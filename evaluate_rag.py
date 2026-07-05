import os
import time
import warnings
from dotenv import load_dotenv

# -------------------------------------------------
# ① ENVIRONMENT & LANGSMITH SETUP
# -------------------------------------------------
# WHY: load_dotenv() reads your .env file which contains:
#   - OPENROUTER_API_KEY (for LLM calls)
#   - LANGSMITH_TRACING=true (enables automatic trace logging)
#   - LANGSMITH_API_KEY (authenticates with LangSmith)
#   - LANGSMITH_PROJECT (where traces appear in dashboard)
#
# When LANGSMITH_TRACING=true, EVERY LangChain LLM call
# (both your RAG chain AND RAGAS internal evaluator calls)
# gets automatically sent to smith.langchain.com
# You don't write any extra code — it's automatic!
os.environ["USER_AGENT"] = "Mozilla/5.0"
load_dotenv()

# ragas.evaluate() is deprecated in favor of the @experiment() decorator in 0.4.x,
# but it's still fully functional (just emits a DeprecationWarning on every call).
# Silencing only that specific warning so it doesn't clutter the output.
warnings.filterwarnings("ignore", message=".*evaluate\\(\\) is deprecated.*")

# Verify LangSmith is configured
print("=" * 60)
print("🔧 LANGSMITH CONFIGURATION")
print("=" * 60)
tracing = os.environ.get("LANGSMITH_TRACING", "not set")
project = os.environ.get("LANGSMITH_PROJECT", "not set")
api_key = os.environ.get("LANGSMITH_API_KEY", "not set")
print(f"  Tracing enabled : {tracing}")
print(f"  Project name    : {project}")
print(f"  API key set     : {'✅ Yes' if api_key != 'not set' else '❌ No'}")
print()

# -------------------------------------------------
# ② IMPORTS
# -------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# pyrefly: ignore [missing-import]
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# CrossEncoderReranker lives in langchain_classic now (moved out of core `langchain`
# in the langchain-classic split). pip install langchain-classic if this fails.
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from collections import defaultdict
import pickle

# RAGAS imports (ragas 0.4.x collections-based metrics API)
#
# KNOWN UPSTREAM BUG (ragas==0.4.3, still unfixed as of this writing):
# ragas/llms/base.py has a hardcoded `from langchain_community.chat_models.vertexai
# import ChatVertexAI`. That submodule was removed from langchain-community (Vertex AI
# support moved to the standalone langchain-google-vertexai package), so on any
# reasonably current langchain-community install, `from ragas import ...` crashes with:
#   ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
# even though we never touch Vertex AI anywhere in this script.
#
# Fix: pre-populate sys.modules with a stand-in module BEFORE importing ragas, so its
# internal import finds something and doesn't try to hit the real (missing) path.
# We use a dummy placeholder class instead of the real langchain_google_vertexai
# ChatVertexAI so this doesn't force you to install a Vertex AI dependency you don't use.
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    class _ChatVertexAIStub:
        """Placeholder so ragas's internal isinstance()/import checks succeed.
        Never actually instantiated since we don't use Vertex AI."""
        pass

    _vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_shim.ChatVertexAI = _ChatVertexAIStub
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim

from openai import AsyncOpenAI
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory

print("✅ All imports loaded")

# -------------------------------------------------
# ③ MAIN LLM (used for RAG generation + multi-query expansion)
# -------------------------------------------------
# This is the same model as in Hybrid_RAG.py. It's a LangChain chat model,
# used only for the retrieval/generation chain below — NOT for RAGAS scoring.
# The RAGAS evaluator LLM is set up separately in section ⑫, because ragas 0.4.x
# needs a native OpenAI-SDK-style async client (via llm_factory), not a LangChain
# chat model wrapper.
main_llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0,
    max_tokens=1024
)

print("✅ Main LLM (gpt-4o-mini via OpenRouter) loaded")

# -------------------------------------------------
# ④ EMBEDDINGS
# -------------------------------------------------
# Same embedding model as Hybrid_RAG.py
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embedding model loaded")

# -------------------------------------------------
# ⑤ LOAD INDEXES (same as Hybrid_RAG.py)
# -------------------------------------------------
FAISS_INDEX_PATH = "faiss_index_pdf"
BM25_INDEX_PATH = "bm25_index_pdf.pkl"

print("⏳ Loading FAISS index...")
vectorstore = FAISS.load_local(
    folder_path=FAISS_INDEX_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
print("✅ FAISS index loaded")

print("⏳ Loading BM25 index...")
with open(BM25_INDEX_PATH, "rb") as f:
    sparse_retriever = pickle.load(f)
print("✅ BM25 index loaded")

# -------------------------------------------------
# ⑥ CREATE RETRIEVERS (same as Hybrid_RAG.py)
# -------------------------------------------------
dense_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5}
)
sparse_retriever.k = 20
print("✅ Dense & Sparse retrievers ready")

# -------------------------------------------------
# ⑦ CROSS-ENCODER RERANKER (same as Hybrid_RAG.py)
# -------------------------------------------------
print("⏳ Loading CrossEncoder Reranker...")
cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=8)
print("✅ CrossEncoder Reranker ready")

# -------------------------------------------------
# ⑧ RRF + MULTI-QUERY (same logic as Hybrid_RAG.py)
# -------------------------------------------------
def reciprocal_rank_fusion(results, k=10):
    fused_scores = defaultdict(float)
    doc_map = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_id = str(doc.metadata.get("page", "")) + "_" + str(hash(doc.page_content))
            doc_map[doc_id] = doc
            fused_scores[doc_id] += 1.0 / (rank + k)
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in reranked]

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
    response = (query_prompt | main_llm | StrOutputParser()).invoke({"question": question})
    queries = [q.strip() for q in response.split('\n') if q.strip()]
    return queries

def hybrid_retriever(query):
    """Full hybrid retrieval: Multi-Query → Dense + Sparse → RRF → CrossEncoder Rerank"""
    alt_queries = generate_queries(query)
    all_queries = [query] + alt_queries

    all_results = []
    for q in all_queries:
        all_results.append(dense_retriever.invoke(q))
        all_results.append(sparse_retriever.invoke(q))

    fused_docs = reciprocal_rank_fusion(all_results)
    top_50 = fused_docs[:50]
    reranked_docs = reranker.compress_documents(top_50, query)
    return reranked_docs

print("✅ Hybrid Retriever ready")

# -------------------------------------------------
# ⑨ RAG CHAIN (same as Hybrid_RAG.py)
# -------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.

Synthesize ONLY using the retrieved context. Answer the user's question as completely as possible using the provided information.

If the answer is not present, say you don't know."""),
    ("human", """Context:
{context}

Question:
{question}""")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": RunnableLambda(hybrid_retriever) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | main_llm
    | StrOutputParser()
)
print("✅ RAG Chain ready")

# -------------------------------------------------
# ⑩ EVALUATION DATASET — GROUND TRUTH FROM THE PDF
# -------------------------------------------------
# WHY GROUND TRUTH?
# RAGAS needs a "reference" (the correct answer) for Context Recall.
# Context Recall checks: "Does the retrieved context contain ALL the
# information needed to produce the reference answer?"
#
# HOW TO READ THIS:
# Each entry has:
#   - user_input: The question we ask
#   - reference: The CORRECT answer (we wrote this from the PDF)
#   - retrieved_contexts: EMPTY for now — we'll fill it by running the retriever
#   - response: EMPTY for now — we'll fill it by running the RAG chain

print("\n" + "=" * 60)
print("📋 EVALUATION DATASET (8 questions from the PDF)")
print("=" * 60)

eval_questions = [
    # ── Chapter 1: Getting Started (Polynomial Fitting & Web Traffic) ──
    {
        "user_input": "How was the error reduced from 317389767.34 to 179983507.878?",
        "reference": (
            "The error was reduced from 317,389,767.34 to 179,983,507.878 by using a polynomial model "
            "of degree 2 instead of a straight line (degree 1). Using sp.polyfit(x, y, 1) gives a "
            "straight line with error 317,389,767.34, while sp.polyfit(x, y, 2) fits a parabola with "
            "error 179,983,507.878, which is almost half the error of the straight line model."
        ),
    },
    {
        "user_input": "Why is polyfit() used when fitting a straight line to web traffic data?",
        "reference": (
            "polyfit() is used because it finds the best coefficients of a polynomial that minimize "
            "the residual error. For a straight line (order 1), polyfit(x, y, 1) returns two "
            "coefficients representing y = 2596.19x + 109532.17. The error function used inside "
            "polyfit is the residual of the fitted polynomial: error = sum((y' - y)^2), and polyfit "
            "finds the coefficients such that this error function is minimized."
        ),
    },
    {
        "user_input": "How are NaN values removed before plotting web traffic data?",
        "reference": (
            "NaN values are removed using NumPy's isnan() function combined with boolean indexing. "
            "First, sp.isnan(y) creates a boolean array marking invalid values. Then the negation "
            "operator ~ is used: x = x[~sp.isnan(y)] and y = y[~sp.isnan(y)]. This filters out all "
            "data points where y has invalid (NaN) values, keeping only valid entries."
        ),
    },
    {
        "user_input": "Why does NumPy's array summation perform faster than Python loops?",
        "reference": (
            "NumPy's array operations are faster because NumPy delegates all its work to highly "
            "optimized C and Fortran code under the hood. For example, summing a NumPy array of "
            "1,000,000 elements is about 67 times faster than using Python's built-in sum() on a "
            "normal list. This speed difference adds up significantly in machine learning workloads."
        ),
    },
    {
        "user_input": "What is overfitting and why is a degree 50 polynomial problematic?",
        "reference": (
            "Overfitting occurs when a model is too complex and fits the training data too closely, "
            "capturing noise instead of the underlying pattern. A polynomial of degree 50 nicely "
            "passes through all data points from the past and has a low training error of 109,504,587.153, "
            "but it will most probably not provide good results for future requests. The model captures "
            "the noise in the data rather than the true underlying process."
        ),
    },
    {
        "user_input": "What is the purpose of train/test split in machine learning?",
        "reference": (
            "The purpose of train/test split is to properly evaluate a model's performance on unseen data. "
            "If we measure the error of a model on the same data used for training, the error will be "
            "misleadingly low. By splitting data into training and test datasets, we can train the model "
            "on one portion and evaluate it on another. The training error is not a good indicator of "
            "how well the model performs on unseen data — the test error is more reliable."
        ),
    },
    {
        "user_input": "How is web traffic data read from a file using SciPy?",
        "reference": (
            "Web traffic data is read using SciPy's genfromtxt() function: "
            "data = sp.genfromtxt('web_traffic.tsv', delimiter='\\t'). The file is tab-separated (.tsv). "
            "The data is then split into two arrays: x = data[:,0] for the hours and y = data[:,1] "
            "for the web hits. There are 8 hours out of 743 total that contain NaN (invalid) values "
            "which need to be cleaned before analysis."
        ),
    },
    {
        "user_input": "What are the errors for polynomial degrees 3, 10, and 50 when fitting web traffic data?",
        "reference": (
            "The errors for increasing polynomial degrees are: degree 3 has error 139,350,144.032, "
            "degree 10 has error 121,942,326.869, and degree 50 has error 109,504,587.153. While "
            "higher degree polynomials have lower training errors, they overfit the data. The degree 2 "
            "polynomial with error 179,983,507.878 provides a better balance between fitting the data "
            "and generalizing to future data."
        ),
    },
    # ── Chapter 1: Loss Functions & Model Selection ──
    {
        "user_input": "What error function does polyfit use internally and why not use absolute values?",
        "reference": (
            "The error function used inside polyfit is the residual: error = sum((y' - y)^2), where "
            "y' is the predicted value and y is the actual value. Squaring is used instead of absolute "
            "values because the absolute function introduces non-differentiable points, which makes "
            "the optimization problem harder. Squaring provides a smooth, differentiable function "
            "that standard optimization algorithms can efficiently minimize."
        ),
    },
    {
        "user_input": "What is the bias-variance trade-off in machine learning?",
        "reference": (
            "The bias-variance trade-off is a fundamental concept. Bias means a model makes strong "
            "assumptions and may miss important patterns (e.g., fitting a straight line to parabolic data). "
            "Variance means a model is too sensitive to training data and captures noise (e.g., degree 50 "
            "polynomial). The total error decomposes as: Total Error = Bias squared + Variance + "
            "Irreducible Noise. The sweet spot is a model complex enough to capture the true pattern "
            "but not so complex that it memorizes noise, like the degree 2 polynomial."
        ),
    },
    # ── Chapter 2: Classification ──
    {
        "user_input": "What is cross-validation and how does five-fold cross-validation work?",
        "reference": (
            "Cross-validation is a technique where data is split into x folds (groups). In five-fold "
            "cross-validation, you learn five models, each time leaving one fold (20 percent) out of "
            "the training data. Each model is tested on the left-out fold and the results are averaged. "
            "This provides most of the benefits of leave-one-out validation at a fraction of the cost. "
            "At the end, you train a final model on all your data, and the cross-validation loop gives "
            "you an estimate of how well this model should generalize."
        ),
    },
    # ── Chapter 3: Clustering ──
    {
        "user_input": "What is the difference between supervised and unsupervised learning?",
        "reference": (
            "In supervised learning, the learning is guided by a teacher in the form of correct "
            "classifications or labels. You learn a model from training data paired with their "
            "respective classes. In unsupervised learning, you do not possess labels to guide the "
            "learning. You have to learn structure from the data alone. Clustering is an example "
            "of unsupervised learning where similar items are grouped into clusters without "
            "predefined labels."
        ),
    },
    {
        "user_input": "What is TF-IDF and how is it calculated?",
        "reference": (
            "TF-IDF (term frequency - inverse document frequency) is a measure of how important "
            "a word is to a document. It increases the weight of words specific to a particular "
            "document while decreasing the weight of words that appear across many documents. "
            "It is calculated as tfidf(t, d) = tf(t, d) * idf(t), where tf(t, d) is the term "
            "frequency (how often word t appears in document d) and idf(t) = log(|D| / (1 + df(t))) "
            "is the inverse document frequency, where |D| is total documents and df(t) is the "
            "number of documents containing term t."
        ),
    },
    {
        "user_input": "How does KMeans clustering work for finding related posts?",
        "reference": (
            "KMeans clustering groups similar posts together. Using scikit-learn: km = KMeans("
            "n_clusters=50, n_init=1) and km.fit(X_train). After fitting, every post is assigned "
            "to one of the clusters. For a new post, you transform it using the vectorizer, predict "
            "its cluster with km.predict(new_post_vec), then find similar posts within the same "
            "cluster using similar_indices = (km.labels_ == new_post_label).nonzero()[0]. This "
            "two-step approach is much faster than computing similarity to every single post."
        ),
    },
    {
        "user_input": "What is stemming and how does NLTK's SnowballStemmer work?",
        "reference": (
            "Stemming is the heuristic of chopping off the end of a word to transform it into "
            "its root form. Using NLTK's SnowballStemmer: s = nltk.stem.SnowballStemmer('english'). "
            "It supports 13 languages. For example, s.stem('graphics') returns 'graphic', "
            "s.stem('imaging') returns 'imag', and s.stem('image') also returns 'imag'. "
            "However, stemming is not perfect — 'imagination' stems to 'imagin' which is a "
            "different root than 'imag'."
        ),
    },
    # ── Chapter 4: Sentiment Analysis ──
    {
        "user_input": "How does Naive Bayes classification work for text data?",
        "reference": (
            "Naive Bayes uses Bayes' theorem with the naive assumption that features are independent. "
            "P(class|features) = P(features|class) * P(class) / P(features). The naive part assumes "
            "P(features|class) = P(feature1|class) * P(feature2|class) * ... During training, it "
            "counts how often each word appears in each class and calculates P(word|class). During "
            "classification, it multiplies the probability of each word given each class, multiplies "
            "by the prior, and picks the class with the highest product. MultinomialNB from "
            "scikit-learn is used for text classification."
        ),
    },
    {
        "user_input": "What is the difference between Naive Bayes and Logistic Regression for text classification?",
        "reference": (
            "Naive Bayes is a generative model that estimates P(features|class) using the naive "
            "independence assumption, while Logistic Regression is a discriminative model that "
            "directly models the decision boundary. Logistic regression fits a linear model to the "
            "log-odds: log(P(y=1|x) / P(y=0|x)) = w*x + b. It has a regularization parameter C — "
            "smaller C means stronger regularization. Logistic regression typically achieves better "
            "accuracy than Naive Bayes but is slower to train. Feature engineering often matters "
            "more than the choice of classifier."
        ),
    },
    # ── Chapter 7: Regression ──
    {
        "user_input": "What is the difference between Lasso, Ridge, and ElasticNet regression?",
        "reference": (
            "Lasso (L1 penalty), Ridge (L2 penalty), and ElasticNet (both L1 and L2) are penalized "
            "regression models. Both Lasso and Ridge result in smaller coefficients than unpenalized "
            "regression. However, the Lasso has the additional property that it results in many "
            "coefficients being set to exactly zero, meaning the final model does not even use some "
            "of its input features, effectively performing feature selection. ElasticNet combines "
            "both penalties."
        ),
    },
    # ── Chapter 8: Recommendations ──
    {
        "user_input": "What are the two main approaches to recommendation systems?",
        "reference": (
            "The two main approaches are: 1) Content-based filtering, which recommends items similar "
            "to what the user has liked before based on item features, and 2) Collaborative filtering, "
            "which recommends items that similar users have liked based on user behavior patterns. "
            "Normalization is critical for recommendations because different users have different "
            "rating scales. A generous user might rate everything 4-5 while a strict user gives 2-3 "
            "for the same quality. Normalizing by subtracting each user's mean rating makes "
            "comparisons meaningful."
        ),
    },
    # ── Chapter 8: Neural Networks ──
    {
        "user_input": "What are the main components of an artificial neural network?",
        "reference": (
            "An artificial neural network consists of interconnected nodes (neurons) organized in "
            "layers: 1) Input layer that receives the raw features, 2) Hidden layers that transform "
            "the features through weighted connections, and 3) Output layer that produces the final "
            "prediction. Each connection has a weight, and each neuron applies an activation function "
            "to its weighted sum of inputs. Learning consists of adjusting these weights to minimize "
            "a loss function."
        ),
    },
]

# Limit to 8 questions for faster evaluation
eval_questions = eval_questions[:8]

print(f"  Created {len(eval_questions)} evaluation questions\n")
for i, q in enumerate(eval_questions, 1):
    print(f"  Q{i}: {q['user_input'][:70]}...")

# -------------------------------------------------
# ⑪ RUN THE RAG PIPELINE — FILL IN RESPONSES & CONTEXTS
# -------------------------------------------------
# WHY: RAGAS needs 4 fields per sample:
#   1. user_input    ✅ (already have — the question)
#   2. reference     ✅ (already have — ground truth from PDF)
#   3. retrieved_contexts ❌ (need to run retriever)
#   4. response      ❌ (need to run RAG chain)
#
# This step runs your actual Hybrid RAG pipeline for each question
# and captures what it retrieves + what it answers.
# Each of these calls gets traced to LangSmith automatically!

import json

print("\n" + "=" * 60)
print("🚀 RUNNING RAG PIPELINE FOR EACH QUESTION")
print("=" * 60)

CACHE_FILE = "rag_cache.json"
FORCE_REGENERATE = False  # Set this to True when you change top_k, embeddings, etc!

eval_samples = []

if not FORCE_REGENERATE and os.path.exists(CACHE_FILE):
    print(f"📦 Found existing cache file '{CACHE_FILE}'. Loading saved answers...")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cached_data = json.load(f)
    for d in cached_data:
        eval_samples.append(SingleTurnSample(
            user_input=d["user_input"],
            reference=d["reference"],
            retrieved_contexts=d["retrieved_contexts"],
            response=d["response"],
        ))
    print(f"✅ Loaded {len(eval_samples)} samples from cache. (Set FORCE_REGENERATE=True to overwrite)")
else:
    if FORCE_REGENERATE:
        print("⚠️ FORCE_REGENERATE is True. Ignoring cache and re-running RAG pipeline...")
    else:
        print(f"⚠️ Cache file '{CACHE_FILE}' not found. Running RAG pipeline for the first time...")
        
    for i, q in enumerate(eval_questions, 1):
        question = q["user_input"]
        reference = q["reference"]

        print(f"\n--- Question {i}/{len(eval_questions)} ---")
        print(f"Q: {question[:80]}...")

        # Step A: Run the hybrid retriever to get contexts
        print("  ⏳ Retrieving contexts...")
        start = time.time()
        docs = hybrid_retriever(question)
        retrieved_contexts = [doc.page_content for doc in docs]
        print(f"  ✅ Retrieved {len(retrieved_contexts)} chunks ({time.time()-start:.1f}s)")

        # Step B: Run the RAG chain to get the answer
        print("  ⏳ Generating answer...")
        start = time.time()
        response = rag_chain.invoke(question)
        print(f"  ✅ Got answer ({time.time()-start:.1f}s)")
        print(f"  A: {response[:100]}...")

        # Step C: Create a SingleTurnSample with all 4 fields filled
        sample = SingleTurnSample(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
            response=response,
        )
        eval_samples.append(sample)

    # Save to cache
    cached_data = [
        {
            "user_input": s.user_input,
            "reference": s.reference,
            "retrieved_contexts": s.retrieved_contexts,
            "response": s.response
        }
        for s in eval_samples
    ]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cached_data, f, indent=4)
    print(f"\n💾 Saved generated answers to '{CACHE_FILE}' for future evaluations!")

print(f"\n✅ All {len(eval_samples)} samples collected!")

# -------------------------------------------------
# ⑫ SET UP THE RAGAS EVALUATOR (LLM + EMBEDDINGS)
# -------------------------------------------------
# WHY AsyncOpenAI, NOT OpenAI:
# ragas 0.4.x metrics (ragas.metrics.collections) are async under the hood —
# evaluate() calls each metric's .ascore(), which calls the LLM's .agenerate().
# A synchronous client will raise:
#   TypeError: Cannot use agenerate() with a synchronous client. Use generate() instead.
# so this MUST be AsyncOpenAI, not OpenAI.
#
# We point it at OpenRouter's OpenAI-compatible endpoint and use gpt-4o-mini
# as the judge (fast, cheap, high rate limits).

print("\n" + "=" * 60)
print("🔧 SETTING UP RAGAS EVALUATOR")
print("=" * 60)

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    timeout=300.0,
    max_retries=5,
)

# llm_factory() wraps the client with ragas's structured-output layer (Instructor).
ragas_llm = llm_factory(
    "meta/llama-3.1-70b-instruct",
    client=nvidia_client,
    temperature=0.7,
    top_p=0.7,
    max_tokens=1024,
)

print("  ✅ Wrapped Llama-3.1-70B (via NVIDIA API) as RAGAS evaluator LLM")

# -------------------------------------------------
# ⑬ DEFINE RAGAS METRICS
# -------------------------------------------------
# IMPORTANT (ragas 0.4.x): collections-based metrics require the LLM (and, for
# AnswerRelevancy, the embeddings) to be bound AT CONSTRUCTION TIME — not just
# passed into evaluate() afterwards. Faithfulness(), AnswerRelevancy(), etc. with
# no arguments will fail validation since `llm` is a required field.
#
# HOW EACH METRIC WORKS INTERNALLY:
#
# ┌─ Faithfulness ────────────────────────────────────────────────┐
# │ 1. Breaks the answer into individual claims/statements        │
# │ 2. For each claim, asks LLM: "Is this supported by context?" │
# │ 3. Score = (supported claims) / (total claims)                │
# │ Example: Answer has 5 claims, 4 are in context → 0.80        │
# └───────────────────────────────────────────────────────────────┘
#
# ┌─ Answer Relevancy ────────────────────────────────────────────┐
# │ 1. LLM generates hypothetical questions from the answer       │
# │ 2. Computes embedding similarity between generated            │
# │    questions and the original question                        │
# │ 3. High similarity → answer is relevant to the question       │
# └───────────────────────────────────────────────────────────────┘
#
# ┌─ Context Precision ──────────────────────────────────────────-┐
# │ 1. For each retrieved chunk, LLM judges: "Is this relevant    │
# │    to answering the question given the reference?"            │
# │ 2. Checks if relevant chunks appear BEFORE irrelevant ones    │
# │ 3. Uses a weighted precision formula (top-ranked = more imp.) │
# └───────────────────────────────────────────────────────────────┘
#
# ┌─ Context Recall ─────────────────────────────────────────────-┐
# │ 1. Breaks the reference answer into individual statements     │
# │ 2. For each statement, checks if ANY retrieved chunk           │
# │    contains that information                                  │
# │ 3. Score = (statements found in context) / (total statements) │
# └───────────────────────────────────────────────────────────────┘

metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=embeddings),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

print("  ✅ Metrics configured:")
for m in metrics:
    print(f"      - {m.name}")

# -------------------------------------------------
# ⑭ CREATE DATASET & RUN EVALUATION
# -------------------------------------------------
# This is where the magic happens!
# RAGAS will:
#   1. Take each sample (question, context, response, reference)
#   2. Run each metric against each sample
#   3. Each metric makes internal LLM calls (to gpt-4o-mini via OpenRouter)
#   4. All LLM calls are traced to LangSmith automatically
#   5. Return scores for each metric

print("\n" + "=" * 60)
print("📊 RUNNING RAGAS EVALUATION")
print("=" * 60)
print("  This will take a few minutes...")
print("  (Each metric makes LLM calls to judge quality)")
print("  (All calls are being traced to LangSmith!)\n")

dataset = EvaluationDataset(samples=eval_samples)

start_time = time.time()
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ragas_llm,
    run_config=RunConfig(max_workers=1, timeout=300)
)
elapsed = time.time() - start_time

# -------------------------------------------------
# ⑮ DISPLAY RESULTS
# -------------------------------------------------
print("\n" + "=" * 60)
print("📊 RAGAS EVALUATION RESULTS")
print("=" * 60)

# Overall scores
print("\n🏆 OVERALL SCORES:")
print("-" * 40)

# EvaluationResult -> pandas DataFrame, one row per sample, one column per metric.
df = results.to_pandas()
metric_names = [m.name for m in metrics]

import math
for metric_name in metric_names:
    if metric_name in df.columns:
        score = df[metric_name].mean()
        if math.isnan(score):
            print(f"  {metric_name:25s} {'[Calculation Failed]':<21} NaN")
        else:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric_name:25s} {bar} {score:.4f}")

# Per-question breakdown
print("\n\n📋 PER-QUESTION BREAKDOWN:")
print("-" * 80)

print(df.to_string(index=False))

# Save results to CSV
csv_path = "ragas_evaluation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n💾 Results saved to: {csv_path}")

# -------------------------------------------------
# ⑯ INTERPRETATION GUIDE
# -------------------------------------------------
print("\n" + "=" * 60)
print("📖 HOW TO INTERPRET YOUR SCORES")
print("=" * 60)
print("""
┌──────────────────────┬────────────────────────────────────────────┐
│ Score Range           │ Interpretation                            │
├──────────────────────┼────────────────────────────────────────────┤
│ 0.9 - 1.0            │ 🟢 Excellent — your RAG is performing     │
│                      │    very well on this dimension             │
├──────────────────────┼────────────────────────────────────────────┤
│ 0.7 - 0.9            │ 🟡 Good — room for improvement            │
│                      │    but acceptable for most use cases       │
├──────────────────────┼────────────────────────────────────────────┤
│ 0.5 - 0.7            │ 🟠 Mediocre — needs attention              │
│                      │    Consider tuning retrieval/prompts       │
├──────────────────────┼────────────────────────────────────────────┤
│ 0.0 - 0.5            │ 🔴 Poor — significant issues               │
│                      │    Major changes needed                    │
└──────────────────────┴────────────────────────────────────────────┘

IF FAITHFULNESS IS LOW:
  → Your LLM is hallucinating (making up info not in the context)
  → Fix: Make the system prompt stricter, lower temperature

IF ANSWER RELEVANCY IS LOW:
  → Your LLM is not answering what was asked
  → Fix: Improve the prompt template, ensure question is clear

IF CONTEXT PRECISION IS LOW:
  → Your retriever is returning irrelevant chunks at the top
  → Fix: Tune MMR lambda, increase reranker top_n, improve chunks

IF CONTEXT RECALL IS LOW:
  → Your retriever is missing important information
  → Fix: Increase k, reduce chunk_size, improve embeddings
""")

# -------------------------------------------------
# ⑰ LANGSMITH INSTRUCTIONS
# -------------------------------------------------
print("=" * 60)
print("🔗 CHECK RESULTS IN LANGSMITH")
print("=" * 60)
print(f"""
1. Open: https://smith.langchain.com
2. Click on project: "{project}"
3. You'll see traces for:
   - Each RAG chain call (retrieval + generation)  
   - Each RAGAS metric computation (internal LLM calls)
4. Click any trace to see:
   - Full execution tree (retriever → reranker → LLM)
   - Input/output at each step
   - Latency breakdown
   - Token usage

⏱️  Evaluation completed in {elapsed:.1f} seconds
""")

print("✅ RAGAS Evaluation Complete!")