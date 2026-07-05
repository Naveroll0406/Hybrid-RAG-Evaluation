import os
from dotenv import load_dotenv

print("✅ Starting RAG Pipeline...")

# ----------------------------------
# Set USER_AGENT
# ----------------------------------
os.environ["USER_AGENT"] = "Mozilla/5.0"
print("✅ USER_AGENT configured")

# ----------------------------------
# Imports
# ----------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np
# pyrefly: ignore [missing-import]
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("✅ Libraries imported")

# ----------------------------------
# Load environment variables
# ----------------------------------
load_dotenv()
print("✅ Environment variables loaded")

# ----------------------------------
# LLM
# ----------------------------------
print("⏳ Initializing LLM...")
llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0
)
print("✅ LLM initialized")

# ----------------------------------
# Embeddings
# ----------------------------------
print("⏳ Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embedding model loaded")

# ----------------------------------
# Vector Store Persistence with FAISS
# ----------------------------------
FAISS_INDEX_PATH = "faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    print("⏳ Loading existing FAISS index from disk...")
    # Loading requires allow_dangerous_deserialization=True for FAISS
    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.COSINE
    )
    print("✅ FAISS index loaded from disk (skipped PDF processing and embedding)")
else:
    # ----------------------------------
    # Load PDF
    # ----------------------------------
    print("⏳ Loading PDF...")
    loader = PyPDFLoader("1706.03762v7.pdf")
    docs = loader.load()
    
    print(f"✅ PDF loaded ({len(docs)} pages)")
    
    # ----------------------------------
    # Split Documents
    # ----------------------------------
    print("⏳ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    
    splits = splitter.split_documents(docs)
    
    print(f"✅ Documents split into {len(splits)} chunks")
    
    # ----------------------------------
    # Create and Save Vector Store (FlatIVF)
    # ----------------------------------
    print("⏳ Creating FAISS vector store with FlatIVF index...")
    
    # 1. Embed all documents first to train the IVF index
    texts = [doc.page_content for doc in splits]
    print(f"⏳ Generating embeddings to train FlatIVF index for {len(texts)} chunks...")
    emb_list = embeddings.embed_documents(texts)
    emb_np = np.array(emb_list, dtype=np.float32)
    faiss.normalize_L2(emb_np) # Normalize for Cosine Similarity
    
    # 2. Setup FlatIVF Index
    dimension = emb_np.shape[1]
    nlist = max(1, len(texts) // 4) # heuristic for nlist based on chunks
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # 3. Train Index
    print(f"⏳ Training IVF index with nlist={nlist}...")
    index.train(emb_np)
    
    # 4. Initialize Langchain FAISS and add documents
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.COSINE
    )
    
    print("⏳ Adding documents to vector store...")
    vectorstore.add_documents(splits)
    
    print("⏳ Saving FAISS index to disk...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index saved to disk for future runs")

# ----------------------------------
# Retriever
# ----------------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

print("✅ Retriever created")

# ----------------------------------
# Prompt
# ----------------------------------
prompt = ChatPromptTemplate.from_template(
    """
Use the following context to answer the question.

If you don't know, say you don't know.

Context:
{context}

Question:
{question}
"""
)

print("✅ Prompt created")

# ----------------------------------
# Helper
# ----------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("✅ format_docs function created")

# ----------------------------------
# Build Chain
# ----------------------------------
print("⏳ Building RAG chain...")

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain built")

# ----------------------------------
# Question
# ----------------------------------
question = "What is the Transformer architecture and what are its key components?"

print("\n━━━━━━━━━━━━━━━━━━")
print("Question:")
print(question)
print("━━━━━━━━━━━━━━━━━━")

# ----------------------------------
# Retrieve Documents
# ----------------------------------
print("⏳ Retrieving relevant chunks...")
retrieved_docs = retriever.invoke(question)
print(f"✅ Retrieved {len(retrieved_docs)} chunks")

# Uncomment if you want to inspect chunks
# for i, doc in enumerate(retrieved_docs):
#     print(f"\nChunk {i+1}")
#     print(doc.page_content[:300])

# ----------------------------------
# Generate Answer
# ----------------------------------
print("\n⏳ Sending context to LLM...")
print("\nAnswer:\n")

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)

print("\n\n✅ Generation complete")