from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
import json
from prompts import MODERATION_POLICY
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
)
from prompts import SHORT_PROMPT, FOUR_MARK_PROMPT, SEVEN_MARK_PROMPT, detect_intent, META_KEYWORDS
from groq import Groq
_groq_client = Groq(api_key=GROQ_API_KEY)


print("Loading embedding model...")
_embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

print("Loading reranker...")
_reranker = CrossEncoder(RERANKER_MODEL)

print("Loading LLM...")
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL_NAME,
    max_tokens=MAX_NEW_TOKEN,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def load_index():
    pinecone_index = _pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=_embed_model,
    )
    stats = pinecone_index.describe_index_stats()
    print(f"Index loaded. Pinecone reports {stats.total_vector_count} vectors.")
    return index

def retrieve_and_rerank(index, question: str) -> str | None:
    retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
    nodes = retriever.retrieve(question)
    pairs = [(question, node.get_content()) for node in nodes]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)
    top = [node for score, node in ranked[:RERANKER_TOP_N] if score > RERANKER_THRESHOLD]

    if not top:
        return None

    context = "\n\n---\n\n".join([
        f"[Source: {n.metadata.get('filename','?')} | Page: {n.metadata.get('page_no','?')}]\n{n.get_content()}"
        for n in top
    ])
    return context

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

def ask(index, question: str, memory: str = "") -> tuple[str, str | None]:
    mod = check_moderation(question, memory)
    if mod["violation"]:
        print(f"Blocked [{mod['category']}]: {mod['rationale']}")
        return "This assistant is scoped to RGPV academic subjects only. I can't help with that request.", None

    q = question.lower()
    if any(w in q for w in META_KEYWORDS):
        if memory:
            return f"Your last question was: **{memory.split(chr(10))[0].replace('Previous Q: ', '')}**", None
        return "I don't have your previous question in memory.", None

    intent = detect_intent(question)

    context = retrieve_and_rerank(index, question)
    resolved_question = question
    if context is None and memory:
        topic = memory.split("\n")[0].replace("Previous Q: ", "")
        retry_context = retrieve_and_rerank(index, f"{topic} {question}")
        if retry_context is not None:
            context = retry_context
            resolved_question = f"{topic} — {question}"

    memory_block = f"\n\n[Previous Exchange]\n{memory}\n" if memory else ""
    prompt_map = {"short": SHORT_PROMPT, "4mark": FOUR_MARK_PROMPT, "7mark": SEVEN_MARK_PROMPT}

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
        response = _llm.invoke(messages)
        return response.content, resolved_question

    messages = [
        SystemMessage(content=prompt_map[intent]),
        HumanMessage(content=f"{memory_block}<context>\n{context}\n</context>\n\nQuestion: {resolved_question}"),
    ]
    response = _llm.invoke(messages)
    return response.content, resolved_question

if __name__ == "__main__":
    index = load_index()
    q = "What is an operating system? explain in detail"
    print(ask(index, q))