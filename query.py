import chromadb  # delete this line, no longer used
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

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
)
from prompts import SHORT_PROMPT, FOUR_MARK_PROMPT, SEVEN_MARK_PROMPT, detect_intent, META_KEYWORDS

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

def ask(index, question: str, memory: str = "") -> str:
    q = question.lower()
    if any(w in q for w in META_KEYWORDS):
        if memory:
            return f"Your last question was: **{memory.split(chr(10))[0].replace('Previous Q: ', '')}**"
        return "I don't have your previous question in memory."

    intent = detect_intent(question)
    retrieval_query = (
        memory.split("\n")[0].replace("Previous Q: ", "") if (memory and len(question.split()) <= 3)
        else f"{memory} {question}".strip() if memory
        else question
    )
    context = retrieve_and_rerank(index, retrieval_query)
    memory_block = f"\n\n[Previous Exchange]\n{memory}\n" if memory else ""
    prompt_map = {"short": SHORT_PROMPT, "4mark": FOUR_MARK_PROMPT, "7mark": SEVEN_MARK_PROMPT}

    if context is None:
        note = "\n\nNote: nothing relevant was found in your notes for this — answering from general knowledge instead. Verify against your syllabus."
        messages = [
            SystemMessage(content=prompt_map[intent].replace(
                'If context is insufficient, reply: "This topic is not covered in your notes."',
                "No notes context is available. Answer from general knowledge, following the same structure."
            )),
            HumanMessage(content=f"{memory_block}Question: {question}"),
        ]
        response = _llm.invoke(messages)
        return response.content + note

    messages = [
        SystemMessage(content=prompt_map[intent]),
        HumanMessage(content=f"{memory_block}<context>\n{context}\n</context>\n\nQuestion: {question}"),
    ]
    response = _llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    index = load_index()
    q = "What is an operating system? explain in detail"
    print(ask(index, q))