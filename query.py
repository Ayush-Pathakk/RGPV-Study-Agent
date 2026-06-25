import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from langchain_groq import ChatGroq
from langchain.messages import HumanMessage , AIMessage , SystemMessage

from config import (
    PDF_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K,
    MIN_NEW_TOKEN,
    MAX_NEW_TOKEN,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GROQ_MODEL_NAME,
    EMBED_MODEL_NAME,
    GROQ_API_KEY,
)

def load_index():
    """Connect to the existing Chroma collection and load the index."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
 
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
 
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
 
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
 
    print(f"Index loaded. Collection has {chroma_collection.count()} chunks.")
    return index

def retrieve_context(index, question: str) -> str:
    """Retrieve the top-k most relevant chunks for the given question."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(question)
 
    context = "\n\n---\n\n".join([node.get_content() for node in nodes])
    return context


def ask(index, question: str) -> str:
    """Retrieve context and generate an answer using Groq LLM."""
    context = retrieve_context(index, question)
 
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
    )
 
    messages = [
        SystemMessage(content=(
            "You are an expert teaching assistant for RGPV engineering students. "
            "Answer every question in the following structured format, like a detailed 10-mark exam answer.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. Every section heading must be bold and followed by a colon.\n"
            "2. After every heading, ALWAYS start content on the next line. Never write content on the same line as the heading.\n"
            "3. Every section must have detailed content in key points. Never write one-line sections.\n"
            "4. Always use numbered lists for steps, advantages, applications.\n"
            "5. Never shorten the answer. Always write complete, detailed content for every section.\n\n"
            "USE THIS EXACT STRUCTURE:\n\n"
            "**Definition :**\n\n"
            "(2 lines formal definition. switch to a new line: 'In simple words:' followed by 1 line simple explanation.)\n\n"
            "**Why It Is Needed:**\n\n"
            "(Without [topic]: list 2-3 problems. With [topic]: list 2-3 solutions. Each on new line, in a key point format.)\n\n"
            "**Working (Step-by-Step):**\n"
            "(Minimum 5 numbered steps. Each step on its own line with  explanation in a key point form.)\n\n"
            "**Architecture / Block Diagram:**\n"
            "(Describe all components in key points and their connections in detail. Each component on new line.)\n\n"
            "**Types / Modes:**\n"
            "(List each type with name and detailed explanation on new line in numbered key points.)\n\n"
            "**Advantages:**\n"
            "(Minimum 5 numbered advantages. Each on new line with brief explanation.)\n\n"
            "**Applications:**\n"
            "(Minimum 5 real-world applications. Each on new line with one line explanation.)\n\n"
            "**Mnemonics:**\n"
            "(One creative memory trick to remember key points.)\n\n"
            "**Conclusion:**\n"
            "(3 lines summarizing the key takeaway.)\n\n"
            "IMPORTANT: Skip a section only if it is completely irrelevant to the question. "
            "Never skip Definition, Working, Advantages, Applications, and Conclusion. "
            "Use only the provided context. Do not make up information not present in the context. "
            "Always write accordig to the query, thorough answers. Never give short answers."
        )),
        HumanMessage(content=(
            f"Context from study notes:\n{context}\n\n"
            f"Question: {question}"
        )),
    ]
 
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    index = load_index()
    test_question = "What is a main component of cpu ?"
    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer:\n{ask(index, test_question)}")
 