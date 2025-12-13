# CAD-RAG Full Local Execution Script
# Adapted from cad_rag.ipynb for local execution (without Google Colab)
# This script runs the complete CAD-RAG pipeline

import os
import re
import warnings
warnings.filterwarnings('ignore')

# Set script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("CAD-RAG: Content Analysis Detection using RAG")
print("Full Local Execution")
print("=" * 70)

# ============================================================================
# Block 2: Setup Environment (Local adaptation - no Google Colab)
# ============================================================================
print("\n[Block 2] Setting up environment...")

from dotenv import load_dotenv

# Load .env file if it exists
env_path = os.path.join(SCRIPT_DIR, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("Loaded API keys from .env file")
else:
    print("No .env file found - some features may be limited")

# Check for API keys (optional - will work without them for ML part)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
NEO4J_URI = os.environ.get('NEO4J_URI', '')
NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME', '')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '')

has_openrouter_api = bool(OPENROUTER_API_KEY)
has_news_api = bool(NEWS_API_KEY)
has_neo4j = bool(NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD)

print(f"OpenRouter API Key: {'[OK] Available' if has_openrouter_api else '[X] Not configured'}")
print(f"News API Key: {'[OK] Available' if has_news_api else '[X] Not configured'}")
print(f"Neo4j: {'[OK] Available' if has_neo4j else '[X] Not configured'}")

print("Environment setup complete.")

# ============================================================================
# Block 3: Initialize ChromaDB Vector Store (Local storage)
# ============================================================================
print("\n[Block 3] Initializing ChromaDB Vector Store...")

import chromadb

# Use local ChromaDB path instead of Google Drive
CHROMA_PATH = os.path.join(SCRIPT_DIR, "cad_rag_chroma")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create collections
slur_lexicon_collection = chroma_client.get_or_create_collection(
    name="slur_lexicon",
    metadata={"hnsw:space": "cosine"}
)
reclaimed_speech_collection = chroma_client.get_or_create_collection(
    name="reclaimed_speech",
    metadata={"hnsw:space": "cosine"}
)

print(f"ChromaDB initialized at: {CHROMA_PATH}")
print(f"Slur lexicon collection count: {slur_lexicon_collection.count()}")
print(f"Reclaimed speech collection count: {reclaimed_speech_collection.count()}")

# Initialize embedding function and vector stores using HuggingFace embeddings (free, local)
slur_lexicon_db = None
reclaimed_speech_db = None

try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    slur_lexicon_db = Chroma(
        client=chroma_client,
        collection_name="slur_lexicon",
        embedding_function=embedding_function,
    )
    
    reclaimed_speech_db = Chroma(
        client=chroma_client,
        collection_name="reclaimed_speech",
        embedding_function=embedding_function,
    )
    print("LangChain vector stores initialized with HuggingFace embeddings (free, local).")
except Exception as e:
    print(f"Could not initialize LangChain stores: {e}")

# ============================================================================
# Block 4: Populate Knowledge Base with Initial Data
# ============================================================================
print("\n[Block 4] Populating Knowledge Base...")

import pandas as pd
import spacy

# Load spaCy model
print("Loading spaCy en_core_web_lg model...")
nlp = spacy.load("en_core_web_lg")
print("spaCy model loaded successfully.")

# --- 1. Populate Slur Lexicon and Reclaimed Speech Corpus ---
slur_data = {
    'term': ['gloober', 'zorp'],
    'definition': [
        'A derogatory term for immigrants from a specific region, popularized by online hate groups.',
        'A coded term used to insult a political group.'
    ],
    'target_group': ['Immigrants', 'Political Opponents']
}
slur_df = pd.DataFrame(slur_data)

reclaimed_data = {
    'text': ['', ''],
    'source': ['Community Forum A', 'Academic Paper on Linguistics']
}
reclaimed_df = pd.DataFrame(reclaimed_data)

# Add to ChromaDB
if slur_lexicon_collection.count() == 0:
    slur_lexicon_collection.add(
        documents=slur_df['definition'].tolist(),
        metadatas=[{'term': row['term'], 'target': row['target_group']} for _, row in slur_df.iterrows()],
        ids=[f"slur_{i}" for i in range(len(slur_df))]
    )
    print("Added initial data to slur lexicon.")

if reclaimed_speech_collection.count() == 0:
    reclaimed_speech_collection.add(
        documents=reclaimed_df['text'].tolist(),
        metadatas=[{'source': row['source']} for _, row in reclaimed_df.iterrows()],
        ids=[f"reclaimed_{i}" for i in range(len(reclaimed_df))]
    )
    print("Added initial data to reclaimed speech corpus.")

print(f"Slur lexicon count: {slur_lexicon_collection.count()}")
print(f"Reclaimed speech count: {reclaimed_speech_collection.count()}")

# --- 2. Neo4j Knowledge Graph (skip if not configured) ---
neo4j_driver = None
if has_neo4j:
    try:
        from neo4j import GraphDatabase
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        print("Neo4j driver initialized.")
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}")
else:
    print("Skipping Neo4j (not configured)")

# --- 3. News Ingestion (Truly Free using GoogleNews) ---
# Global list to store news if Neo4j is down
recent_news_cache = []

try:
    from GoogleNews import GoogleNews
    googlenews = GoogleNews(lang='en', period='7d')
    
    print("\nFetching news articles (using GoogleNews)...")
    googlenews.search('immigration policy')
    all_articles = googlenews.result()
    
    # Cache for RAG
    recent_news_cache = [f"{a['title']}: {a['desc']}" for a in all_articles[:5]]
    print(f"Fetched {len(all_articles)} articles.")

    if has_neo4j and neo4j_driver:
        print("Ingesting into Neo4j...")
        def ingest_article_to_kg(tx, article):
            tx.run("""
                MERGE (e:Event {url: $url})
                ON CREATE SET e.title = $title, e.summary = $summary, e.date = $date
                """,
                url=article['link'],
                title=article['title'],
                summary=article['desc'],
                date=article['date']
            )
            # Simple entity extraction for demo
            doc = nlp(article['title'])
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP']:
                    tx.run("""
                        MERGE (e:Event {url: $url})
                        MERGE (en:Entity {name: $name})
                        ON CREATE SET en.type = $type
                        MERGE (e)-->(en)
                        """,
                        url=article['link'], name=ent.text, type=ent.label_
                    )
        
        try:
            # Test connection first with short timeout
            with neo4j_driver.session() as session:
                session.run("RETURN 1").single()
            
            # If successful, ingest
            with neo4j_driver.session() as session:
                for article in all_articles[:5]: # Limit to 5 for speed
                    session.execute_write(ingest_article_to_kg, article)
            print(f"Ingested articles into Neo4j.")
            
        except Exception as e:
            print(f"Neo4j connection failed (skipping KG ingestion): {e}")
            has_neo4j = False # Disable flag for rest of script
    else:
        print("Skipping Neo4j ingestion (Not configured)")

except Exception as e:
    print(f"News/KG Error: {e}")

# ============================================================================
# Block 5: Pre-Retrieval Analysis Function
# ============================================================================
print("\n[Block 5] Defining Pre-Retrieval Analysis...")

def pre_retrieval_analysis(text: str):
    """Extracts named entities and potential neologisms from text."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    neologisms = [token.text for token in doc if token.is_oov]
    return {"entities": list(set(entities)), "neologisms": list(set(neologisms))}

print("Pre-retrieval analysis function defined.")

# ============================================================================
# Block 6: RAG Core - Multi-Query Agent (if Google API available)
# ============================================================================
print("\n[Block 6] Setting up RAG Agent...")

agent_executor = None
llm = None

if has_openrouter_api and slur_lexicon_db:
    try:
        from openai import OpenAI
        
        # Initialize OpenRouter client
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        # Create wrapper function for LLM calls
        def call_llm(prompt: str) -> str:
            """Call OpenRouter LLM with prompt (using Hermes 3 - Llama 3.1 405B)."""
            try:
                # Debug: Print ensuring we are making the call
                print(f"DEBUG: Calling OpenRouter with model nousresearch/hermes-3-llama-3.1-405b:free")
                
                response = openrouter_client.chat.completions.create(
                    model="nousresearch/hermes-3-llama-3.1-405b:free",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    extra_body={
                        "HTTP-Referer": "https://localhost:3000",
                        "X-Title": "CAD-RAG Local"
                    }
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"DEBUGGING ERROR: {e}") # Print full error
                if "404" in str(e) and "data policy" in str(e).lower():
                     return "ERROR: OpenRouter 404. Please enable 'Allow Data Collection' in your OpenRouter Privacy Settings (https://openrouter.ai/settings/privacy) to use Free models."
                raise e
        
        # Store as llm wrapper
        llm = call_llm
        
        # Define tool functions (will be called directly instead of via agent)
        def rag_lexical_query(term: str) -> str:
            """Searches the Evolving Slur Lexicon for a definition and origin of a given term."""
            docs = slur_lexicon_db.similarity_search(term, k=1)
            if docs:
                return f"Lexicon Result for '{term}': {docs[0].page_content} (Target: {docs[0].metadata.get('target', 'N/A')})"
            return f"Lexicon Result for '{term}': Term not found."
        
        def rag_contextual_query(entity: str) -> str:
            """Searches the Knowledge Graph for recent news or controversies related to a specific entity."""
            events = []
            if neo4j_driver:
                try:
                    with neo4j_driver.session() as session:
                        result = session.run("""
                            MATCH (e:Event)-->(en:Entity)
                            WHERE toLower(en.name) CONTAINS toLower($entity)
                            RETURN e.title, e.summary
                            ORDER BY e.date DESC
                            LIMIT 2
                            """, entity=entity)
                        events = [f"{record['e.title']}: {record['e.summary']}" for record in result]
                except:
                    pass
            
            # Fallback to simple cache if Neo4j failed or Empty
            if not events and recent_news_cache:
                # Simple keyword match
                events = [n for n in recent_news_cache if entity.lower() in n.lower()]
                
            if events:
                 return f"Contextual Result for '{entity}': Recently mentioned in news: " + " | ".join(events[:2])
            
            return f"Contextual Result for '{entity}': No recent news events found."
        
        def rag_counter_evidence_query(term: str) -> str:
            """Searches the Corpus of Reclaimed Speech for non-hateful uses of a given term."""
            docs = reclaimed_speech_db.similarity_search(term, k=1)
            if docs:
                return f"Counter-Evidence Result for '{term}': Found a potential non-hateful usage: '{docs[0].page_content}'"
            return f"Counter-Evidence Result for '{term}': No instances found in reclaimed speech corpora."
        
        # Create a simple RAG executor function
        def execute_rag_queries(entities: list, neologisms: list) -> str:
            """Execute RAG queries and combine results."""
            results = []
            
            # Query for each entity
            for entity in entities[:3]:  # Limit to 3
                results.append(rag_contextual_query(entity))
            
            # Query for neologisms in lexicon
            for term in neologisms[:3]:  # Limit to 3
                results.append(rag_lexical_query(term))
                results.append(rag_counter_evidence_query(term))
            
            if not results:
                results.append("No specific entities or neologisms found to query.")
            
            return " | ".join(results)
        
        # Store as agent_executor for compatibility
        agent_executor = execute_rag_queries
        
        print("RAG Agent initialized successfully with OpenRouter (free GPT-OSS-120B).")
    except Exception as e:
        print(f"Could not initialize RAG agent: {e}")
else:
    print("Skipping RAG agent (OpenRouter API not configured)")

# ============================================================================
# Block 7: ML Model Training (Using local CSV files)
# ============================================================================
print("\n[Block 7] Training ML Model...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
import numpy as np
import joblib

MODEL_PATH = os.path.join(SCRIPT_DIR, 'cad_rag_model.pkl')
VECTORIZER_PATH = os.path.join(SCRIPT_DIR, 'cad_rag_vectorizer.pkl')

train_file_path = os.path.join(SCRIPT_DIR, 'train.csv')
test_file_path = os.path.join(SCRIPT_DIR, 'traintest.csv')

# Label columns for Jigsaw dataset
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

tfidf_vectorizer = None
logistic_regression_model = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print(f"Loading saved model from: {MODEL_PATH}")
    logistic_regression_model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
elif os.path.exists(train_file_path):
    print(f"Loading training data from: {train_file_path}")
    df_train = pd.read_csv(train_file_path)
    print(f"Training data: {df_train.shape[0]} samples, {df_train.shape[1]} columns")
    
    # Find text column
    text_col = 'comment_text' if 'comment_text' in df_train.columns else 'tweet'
    existing_labels = [col for col in label_columns if col in df_train.columns]
    
    if text_col in df_train.columns and existing_labels:
        # Binarize labels
        for label in existing_labels:
            df_train[label] = (df_train[label] > 0).astype(int)
        
        print(f"\nClass distribution:")
        print(df_train[existing_labels].sum())
        
        # Clean text
        df_train[text_col] = df_train[text_col].fillna('').astype(str)
        df_train[text_col] = df_train[text_col].str.lower()
        df_train[text_col] = df_train[text_col].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
        df_train[text_col] = df_train[text_col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        # Remove NaN
        df_train.dropna(subset=existing_labels, inplace=True)
        
        X_train = df_train[text_col]
        y_train = df_train[existing_labels]
        
        # TF-IDF
        print("\nTraining TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        
        # Train model
        print("Training Logistic Regression model...")
        logistic_regression_model = MultiOutputClassifier(
            LogisticRegression(solver='sag', class_weight='balanced', random_state=42, max_iter=2000)
        )
        logistic_regression_model.fit(X_train_tfidf, y_train)
        print("Model training completed!")
        
        # Save model
        print(f"Saving model to: {MODEL_PATH}")
        joblib.dump(logistic_regression_model, MODEL_PATH)
        joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
        print("Model saved successfully.")
    else:
        print(f"Required columns not found. Available: {list(df_train.columns)}")
else:
    print(f"Training file not found: {train_file_path}")

# Evaluate on test data if available
if os.path.exists(test_file_path) and tfidf_vectorizer and logistic_regression_model:
    print(f"\nLoading test data from: {test_file_path}")
    df_test = pd.read_csv(test_file_path)
    print(f"Test data: {df_test.shape[0]} samples")
    
    # Davidson dataset labels
    davidson_labels = ['hate_speech', 'offensive_language', 'neither']
    test_text_col = 'tweet' if 'tweet' in df_test.columns else 'comment_text'
    existing_davidson = [col for col in davidson_labels if col in df_test.columns]
    
    if test_text_col in df_test.columns:
        for label in existing_davidson:
            df_test[label] = (df_test[label] > 0).astype(int)
        
        df_test[test_text_col] = df_test[test_text_col].astype(str).fillna('')
        X_test = df_test[test_text_col].str.lower()
        X_test = X_test.apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
        X_test = X_test.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        y_test = df_test[existing_davidson].dropna()
        X_test = X_test[y_test.index]
        
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        y_pred_proba = logistic_regression_model.predict_proba(X_test_tfidf)
        probs_positive_class = np.array([p[:, 1] for p in y_pred_proba]).T
        
        # Map predictions
        hate_pred = (probs_positive_class[:, [0, 1, 5]].max(axis=1) >= 0.5).astype(int)
        offensive_pred = (probs_positive_class[:, [2, 4]].max(axis=1) >= 0.5).astype(int)
        neither_pred = ((1 - probs_positive_class.max(axis=1)) >= 0.5).astype(int)
        y_pred_mapped = np.column_stack([hate_pred, offensive_pred, neither_pred])
        
        print("\nCross-Domain Evaluation Metrics:")
        for i, label in enumerate(existing_davidson):
            acc = accuracy_score(y_test.iloc[:, i], y_pred_mapped[:, i])
            prec = precision_score(y_test.iloc[:, i], y_pred_mapped[:, i], zero_division=0)
            rec = recall_score(y_test.iloc[:, i], y_pred_mapped[:, i], zero_division=0)
            f1 = f1_score(y_test.iloc[:, i], y_pred_mapped[:, i], zero_division=0)
            print(f"  {label}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

# ============================================================================
# Block 8: Helper Functions for Classification
# ============================================================================
print("\n[Block 8] Defining Helper Functions...")

def create_dynamic_prompt(original_sentence, context_from_rag, expert_type="General Hate"):
    """Creates the final, context-rich prompt for the expert LLM."""
    PROMPT_TEMPLATE = """You are an expert at detecting {hate_type} hate speech. Your task is to analyze the following sentence based on the provided real-time context and generate a classification (HATEFUL/NOT_HATEFUL) with a detailed rationale.

Sentence: '{sentence}'

Retrieved Context:
{context_block}

Analysis Task: Based ONLY on the sentence and the provided context, classify the sentence and explain your reasoning.
"""
    return PROMPT_TEMPLATE.format(
        hate_type=expert_type,
        sentence=original_sentence,
        context_block=context_from_rag['output'] if isinstance(context_from_rag, dict) and 'output' in context_from_rag else context_from_rag
    )

def determine_sentence_type(sentence: str) -> str:
    """Determines the potential hate speech category."""
    doc = nlp(sentence.lower())
    
    indicators = {
        "Caste-based hate": ["lower castes", "upper castes", "dalit", "brahmin", "caste"],
        "Religion-based hate": ["muslim", "islam", "hindu", "christian", "jew", "sikh", "buddhist", "religion", "allah", "jesus", "mosque", "temple", "church"],
        "Gender-based hate": ["women belong", "men are superior", "feminist", "sexist", "women", "men", "gender"],
        "Political hate": ["political", "government", "traitor", "election", "congress", "regime"],
        "Cyberbullying": ["online", "internet", "dox", "troll", "cyber", "twitter", "facebook"],
        "Subtle hate": ["you know the type", "those kind", "just a joke", "can't take a joke"]
    }
    
    for category, keywords in indicators.items():
        if any(kw in doc.text for kw in keywords):
            return category
    
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP']:
            return "Political hate"
    
    return "General Hate"

def classify_hate_speech_combined(sentence: str) -> dict:
    """Classifies a sentence using the trained model."""
    classifications = {}
    
    if tfidf_vectorizer and logistic_regression_model:
        cleaned = sentence.lower()
        cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        sentence_tfidf = tfidf_vectorizer.transform([cleaned])
        pred_probs = logistic_regression_model.predict_proba(sentence_tfidf)
        probs_pos = np.array([p[:, 1] for p in pred_probs]).T[0]
        
        threshold = 0.5
        predictions = (probs_pos >= threshold).astype(int)
        
        is_hateful = any(predictions)
        classifications['model_result'] = "HATEFUL" if is_hateful else "NOT_HATEFUL"
        classifications['model_labels'] = [label_columns[i] for i, pred in enumerate(predictions) if pred == 1]
        classifications['probabilities'] = dict(zip(label_columns, [f"{p:.3f}" for p in probs_pos]))
    else:
        classifications['model_result'] = "Model not available"
    
    return classifications

print("Helper functions defined.")

# ============================================================================
# Block 9: Main Analysis Function
# ============================================================================
print("\n[Block 9] Defining Main Analysis Function...")

def analyze_sentence(sentence: str):
    """Main analysis function combining all components."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: '{sentence}'")
    print("="*70)
    
    # 1. Pre-Retrieval Analysis
    analysis = pre_retrieval_analysis(sentence)
    print(f"\nEntities found: {analysis['entities']}")
    print(f"Neologisms found: {analysis['neologisms']}")
    
    # 2. Sentence Type
    expert_type = determine_sentence_type(sentence)
    print(f"Sentence category: {expert_type}")
    
    # 3. ML Model Classification
    model_result = classify_hate_speech_combined(sentence)
    print(f"\nML Model Result: {model_result['model_result']}")
    if model_result.get('model_labels'):
        print(f"Detected labels: {model_result['model_labels']}")
    if model_result.get('probabilities'):
        top_probs = sorted(model_result['probabilities'].items(), key=lambda x: float(x[1]), reverse=True)[:3]
        print(f"Top probabilities: {top_probs}")
    
    # 4. RAG Agent (if available)
    retrieved_context = {"output": "RAG agent not available"}
    if agent_executor:
        try:
            # Call RAG with entities and neologisms
            rag_output = agent_executor(analysis['entities'], analysis['neologisms'])
            retrieved_context = {"output": rag_output}
            print(f"\nRAG Context: {rag_output[:200] if len(rag_output) > 200 else rag_output}...")
        except Exception as e:
            print(f"RAG agent error: {e}")
    
    # 5. LLM Rationale (if available)
    llm_rationale = "LLM not available"
    if llm:
        try:
            prompt = create_dynamic_prompt(sentence, retrieved_context, expert_type)
            # Call the LLM wrapper function directly
            llm_rationale = llm(prompt)
            print(f"\nLLM Rationale: {llm_rationale[:300] if len(llm_rationale) > 300 else llm_rationale}...")
        except Exception as e:
            print(f"LLM error: {e}")
    
    return {
        "sentence": sentence,
        "category": expert_type,
        "model_result": model_result,
        "llm_rationale": llm_rationale
    }

print("Main analysis function defined.")

# ============================================================================
# Block 10: Example Usage
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE USAGE - Testing with Sample Sentences")
print("="*70)

test_sentences = [
    "All muslims should be kicked out",
    "Have a great day everyone!",
    "Women belong in the kitchen",
    "You are such an idiot, I hate you",
    "Thank you for your help!",
]

results = []
for sentence in test_sentences:
    result = analyze_sentence(sentence)
    results.append(result)

# Summary
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
for r in results:
    model_result = r['model_result'].get('model_result', 'N/A')
    labels = r['model_result'].get('model_labels', [])
    print(f"\n  '{r['sentence'][:50]}...' -> {model_result}")
    if labels:
        print(f"    Labels: {labels}")

print("\n" + "="*70)
print("CAD-RAG EXECUTION COMPLETED SUCCESSFULLY!")
print("="*70)
