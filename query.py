import json
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import torch
from sentence_transformers import CrossEncoder
from groq import Groq

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBED_MODEL_NAME,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    RERANKER_MODEL,
    RERANKER_THRESHOLD,
    GROQ_MODEL_NAME,
    GROQ_API_KEY,
    MAX_NEW_TOKEN,
    SAFEGUARD_MODEL_NAME,
    CORPUS_PATH,
    HYBRID_CANDIDATE_POOL,
    RRF_K,
)
from hybrid_retrieval import BM25Index, reciprocal_rank_fusion
from prompts import SHORT_PROMPT, FOUR_MARK_PROMPT, SEVEN_MARK_PROMPT, detect_intent, META_KEYWORDS, MODERATION_POLICY

print("Loading embedding model...")
_embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

print("Loading reranker...")
_reranker = CrossEncoder(RERANKER_MODEL, default_activation_function=torch.nn.Sigmoid())

print("Loading BM25 corpus...")
_bm25 = BM25Index(CORPUS_PATH)
if not _bm25.available:
    print("BM25 corpus empty/missing — hybrid search disabled, falling back to dense-only.")

_pc = Pinecone(api_key=PINECONE_API_KEY)
_groq_client = Groq(api_key=GROQ_API_KEY)  # server-side key — moderation only, never used for answers


def load_index():
    pinecone_index = _pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=_embed_model)
    stats = pinecone_index.describe_index_stats()
    print(f"Index loaded. Pinecone reports {stats.total_vector_count} vectors.")
    return index


def identify_key_provider(key: str) -> str:
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("AIza"):
        return "gemini"
    return "unsupported"


def validate_key(key: str, provider: str) -> tuple[bool, str]:
    if provider == "unsupported":
        return False, "That doesn't look like a Groq (gsk_...) or Gemini (AIza...) key."
    try:
        if provider == "groq":
            Groq(api_key=key).models.list()
        else:
            ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=key).invoke("hi")
        return True, "Key verified."
    except Exception:
        return False, f"{provider.capitalize()} rejected this key — check it's active and has quota."


def get_llm(provider: str, api_key: str):
    if provider == "groq":
        return ChatGroq(api_key=api_key, model_name=GROQ_MODEL_NAME, max_tokens=MAX_NEW_TOKEN)
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, max_output_tokens=MAX_NEW_TOKEN)


def retrieve_and_rerank(index, question: str):
    # Dense side (unchanged from before hybrid search)
    retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
    dense_nodes = retriever.retrieve(question)
    dense_candidates = {
        n.node_id: {"text": n.get_content(), "filename": n.metadata.get("filename", "?"), "page_no": n.metadata.get("page_no", "?")}
        for n in dense_nodes
    }

    # Sparse side — no-op if BM25 corpus isn't loaded (dense-only fallback)
    bm25_hits = _bm25.search(question, HYBRID_CANDIDATE_POOL)
    bm25_candidates = {
        r["id"]: {"text": r["text"], "filename": r["filename"], "page_no": r["page_no"]}
        for r in bm25_hits
    }

    # RRF fusion by rank position only — no score-scale mixing
    fused_ids = reciprocal_rank_fusion(
        [list(dense_candidates.keys()), list(bm25_candidates.keys())], k=RRF_K
    )
    all_candidates = {**bm25_candidates, **dense_candidates}  # dense wins on id collision (fresher metadata)
    pool = [(cid, all_candidates[cid]) for cid in fused_ids[:RETRIEVAL_TOP_K]]

    if not pool:
        return None, []

    pairs = [(question, meta["text"]) for _, meta in pool]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)
    top = [meta for score, (cid, meta) in ranked[:RERANKER_TOP_N] if score > RERANKER_THRESHOLD]
    if not top:
        return None, []
    context = "\n\n---\n\n".join([
        f"[Source: {m['filename']} | Page: {m['page_no']}]\n{m['text']}"
        for m in top
    ])
    sources = [{"filename": m["filename"], "page": m["page_no"]} for m in top]
    return context, sources


def summarize_exchange(question: str, answer: str) -> str:
    words = answer.split()[:60]
    short_answer = " ".join(words) + ("..." if len(answer.split()) > 60 else "")
    return f"Previous Q: {question}\nPrevious A (summary): {short_answer}"


def check_moderation(question: str, memory: str = "") -> dict:
    context_note = f"Prior conversation context: {memory}\n\n" if memory else ""
    try:
        response = _groq_client.chat.completions.create(
            model=SAFEGUARD_MODEL_NAME,
            messages=[
                {"role": "system", "content": MODERATION_POLICY},
                {"role": "user", "content": f"{context_note}Content to classify: {question}\n\nAnswer (JSON only):"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "violation": bool(result.get("violation", 0)),
            "category": result.get("category"),
            "rationale": result.get("rationale", ""),
        }
    except Exception as e:
        print(f"Moderation check failed, allowing through: {e}")
        return {"violation": False, "category": None, "rationale": "moderation_error"}


def ask(index, question: str, api_key: str, provider: str, memory: str = "", topic: str = "") -> tuple[str, str | None, str, list]:
    mod = check_moderation(question, memory)
    if mod["violation"]:
        print(f"Blocked [{mod['category']}]: {mod['rationale']}")
        return "This assistant is scoped to RGPV academic subjects only. I can't help with that request.", None, topic, []

    q = question.lower()
    if any(w in q for w in META_KEYWORDS):
        if memory:
            return f"Your last question was: **{memory.split(chr(10))[0].replace('Previous Q: ', '')}**", None, topic, []
        return "I don't have your previous question in memory.", None, topic, []

    intent = detect_intent(question)
    context, sources = retrieve_and_rerank(index, question)
    resolved_question = question
    if context is not None:
        # Fresh direct hit — this question stands on its own, reset the topic anchor.
        topic = question
    elif topic:
        retry_context, retry_sources = retrieve_and_rerank(index, f"{topic} {question}")
        if retry_context is not None:
            context = retry_context
            sources = retry_sources
            resolved_question = f"{topic} — {question}"
            # topic stays as-is: do NOT fold resolved_question back in, or it snowballs across turns.

    memory_block = f"\n\n[Previous Exchange]\n{memory}\n" if memory else ""
    prompt_map = {"short": SHORT_PROMPT, "4mark": FOUR_MARK_PROMPT, "7mark": SEVEN_MARK_PROMPT}
    llm = get_llm(provider, api_key)

    if context is None:
        messages = [
            SystemMessage(content=prompt_map[intent].replace(
                'If context is insufficient, reply: "This topic is not covered in your notes."',
                'No notes context is available for this question. Answer briefly from general '
                'knowledge if it is a reasonable academic question, following the same structure. '
                'If you cannot or should not answer, respond with exactly: '
                '"This assistant is scoped to RGPV academic subjects only." '
                'In either case, end your response with exactly this line on its own: '
                '"_Not verified against your uploaded notes._"'
            )),
            HumanMessage(content=f"{memory_block}Question: {resolved_question}"),
        ]
        response = llm.invoke(messages)
        return response.content, resolved_question, topic, []

    messages = [
        SystemMessage(content=prompt_map[intent]),
        HumanMessage(content=f"{memory_block}<context>\n{context}\n</context>\n\nQuestion: {resolved_question}"),
    ]
    response = llm.invoke(messages)
    return response.content, resolved_question, topic, sources