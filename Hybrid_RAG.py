import os
from dotenv import load_dotenv

# -------------------------------------------------
# Environment
# -------------------------------------------------
os.environ["USER_AGENT"] = "Mozilla/5.0"
load_dotenv()

# -------------------------------------------------
# Imports
# -------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np
from langchain_community.retrievers import BM25Retriever
# pyrefly: ignore [missing-import]
from collections import defaultdict
# pyrefly: ignore [missing-import]
from langchain_openrouter import ChatOpenRouter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# pyrefly: ignore [missing-import]
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
print("✅ Starting Hybrid RAG")

# -------------------------------------------------
# LLM
# -------------------------------------------------
llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0
)

print("✅ LLM loaded")

# -------------------------------------------------
# Embeddings
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("✅ Embedding model loaded")

# -------------------------------------------------
# Setup Index Paths
# -------------------------------------------------
FAISS_INDEX_PATH = "faiss_index_pdf"
BM25_INDEX_PATH = "bm25_index_pdf.pkl"

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(BM25_INDEX_PATH):
    print("⏳ Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded from disk")
    
    print("⏳ Loading existing BM25 index from disk...")
    import pickle
    with open(BM25_INDEX_PATH, "rb") as f:
        sparse_retriever = pickle.load(f)
    print("✅ BM25 index loaded from disk (skipped document processing)")
else:
    # -------------------------------------------------
    # Load PDF
    # -------------------------------------------------
    pdf_path = r"C:\Users\naver\Music\LangChain\RAGs\Building Machine Learning Systems with Python - Second Edition.pdf"
    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    print(f"✅ Loaded {len(docs)} pages")

    # -------------------------------------------------
    # Split
    # -------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400
    )

    splits = splitter.split_documents(docs)

    print(f"✅ Created {len(splits)} chunks")

    # -------------------------------------------------
    # Dense Vector Store
    # -------------------------------------------------
    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )
    
    print("⏳ Saving FAISS index to disk...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index saved")

    # -------------------------------------------------
    # Sparse Vector Store
    # -------------------------------------------------
    sparse_retriever = BM25Retriever.from_documents(splits)
    
    print("⏳ Saving BM25 index to disk...")
    import pickle
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(sparse_retriever, f)
    print("✅ BM25 index saved")

# -------------------------------------------------
# Create Retrievers
# -------------------------------------------------
dense_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "fetch_k": 50,
        "lambda_mult": 0.5
    }
)
sparse_retriever.k = 20 

print("✅ Dense Retriever ready")
print("✅ Sparse Retriever ready")

# -------------------------------------------------
# Cross-Encoder Reranker
# -------------------------------------------------
print("⏳ Loading CrossEncoder Reranker...")
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=8)
print("✅ CrossEncoder Reranker ready")

# -------------------------------------------------
# Reciprocal Rank Fusion (Improved)
# -------------------------------------------------
from collections import defaultdict

def reciprocal_rank_fusion(results, k=10):
    fused_scores = defaultdict(float)
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs):

            # Unique identifier for each chunk
            doc_id = (
                str(doc.metadata.get("page", ""))
                + "_"
                + str(hash(doc.page_content))
            )

            # Store the document
            doc_map[doc_id] = doc

            # Add RRF score
            fused_scores[doc_id] += 1.0 / (rank + k)

    # Sort by descending score
    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return ordered documents
    return [
        doc_map[doc_id]
        for doc_id, _ in reranked
    ]

# -------------------------------------------------
# Multi-Query Generation
# -------------------------------------------------
query_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search. 
Generate variations of the question that format any numbers with commas, and extract just the numbers as standalone keyword queries.
Provide these alternative questions separated by newlines.
Original question: {question}"""
)

def generate_queries(question):
    response = (query_prompt | llm | StrOutputParser()).invoke({"question": question})
    queries = [q.strip() for q in response.split('\n') if q.strip()]
    return queries

# -------------------------------------------------
# Hybrid Retriever (Multi-Query)
# -------------------------------------------------
def hybrid_retriever(query):
    print(f"\n⏳ Generating alternative queries for: {query}")
    alt_queries = generate_queries(query)
    all_queries = [query] + alt_queries
    
    print(f"✅ Generated {len(alt_queries)} alternative queries")
    
    all_results = []
    for q in all_queries:
        all_results.append(dense_retriever.invoke(q))
        all_results.append(sparse_retriever.invoke(q))

    fused_docs = reciprocal_rank_fusion(all_results)

    # Rerank the top 50 fused documents using the CrossEncoder
    top_50 = fused_docs[:50]
    reranked_docs = reranker.compress_documents(top_50, query)

    return reranked_docs

print("✅ Hybrid Retriever (Multi-Query + RRF) ready")

# -------------------------------------------------
# Prompt
# -------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant.

Synthesize ONLY using the retrieved context. Answer the user's question as completely as possible using the provided information.

If the answer is not present, say you don't know.
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

# -------------------------------------------------
# Helper
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# Chain
# -------------------------------------------------
rag_chain = (
    {
        "context": RunnableLambda(hybrid_retriever) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Chain created")

# -------------------------------------------------
# Query
# -------------------------------------------------
# question = "What is the dimensionaltiy reduction and explain in detail?"
# question = "Why is polyfit() used when fitting a straight line?"
# question = "How are NaN values removed before plotting web traffic data?"
# question ="Why does NumPy dot() perform faster than Python loops?"
# question = "What is the value of f2p?"
# question = "almost half the error of the straight line model?"
# question ="179983507.878"
question ="How was the error reduced from 317389767.34 to 179983507.878?"
print("\nQuestion:")
print(question)

# -------------------------------------------------
# Debug Dense
# -------------------------------------------------
dense_docs = dense_retriever.invoke(question)

print("\nDense Results")
for i, doc in enumerate(dense_docs):
    print(
        f"{i+1}. Page {doc.metadata.get('page')} "
        f"- {doc.page_content[:100]}..."
    )

# -------------------------------------------------
# Debug Sparse
# -------------------------------------------------
sparse_docs = sparse_retriever.invoke(question)

print("\nSparse Results")
for i, doc in enumerate(sparse_docs):
    print(
        f"{i+1}. Page {doc.metadata.get('page')} "
        f"- {doc.page_content[:100]}..."
    )

retrieved_docs = hybrid_retriever(question)

print("\nRRF & Reranked Pages:")
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}. Page {doc.metadata.get('page')}")

# -------------------------------------------------
# Answer
# -------------------------------------------------
print("\nAnswer:\n")

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)

print("\n\n✅ Finished")