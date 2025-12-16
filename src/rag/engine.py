# CAD-RAG Engine
# Core logic for Content Analysis Detection using RAG

import os
import re
import warnings
import sys
import pandas as pd
import spacy
import chromadb
import numpy as np
import joblib
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Import expanded hate speech indicators
try:
    from src.features.indicators import HATE_SPEECH_INDICATORS, check_text_for_indicators
except ImportError:
    # Fallback for local testing if src is not in path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.features.indicators import HATE_SPEECH_INDICATORS, check_text_for_indicators

# Set project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Environment
env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
NEO4J_URI = os.environ.get('NEO4J_URI', '')
NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME', '')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', '')

has_openrouter_api = bool(OPENROUTER_API_KEY)
has_news_api = bool(NEWS_API_KEY)
has_neo4j = bool(NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD)

# Initialize ChromaDB Vector Store
CHROMA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "chroma")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

slur_lexicon_collection = chroma_client.get_or_create_collection(name="slur_lexicon", metadata={"hnsw:space": "cosine"})
reclaimed_speech_collection = chroma_client.get_or_create_collection(name="reclaimed_speech", metadata={"hnsw:space": "cosine"})

# Initialize embedding function and vector stores
slur_lexicon_db = None
reclaimed_speech_db = None

try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    slur_lexicon_db = Chroma(client=chroma_client, collection_name="slur_lexicon", embedding_function=embedding_function)
    reclaimed_speech_db = Chroma(client=chroma_client, collection_name="reclaimed_speech", embedding_function=embedding_function)
except Exception as e:
    print(f"Could not initialize LangChain stores: {e}")

# Populate Knowledge Base
nlp = spacy.load("en_core_web_lg")

# Initial Data Loading
def load_initial_data():
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
    
    if slur_lexicon_collection.count() == 0:
        slur_lexicon_collection.add(
            documents=slur_df['definition'].tolist(),
            metadatas=[{'term': row['term'], 'target': row['target_group']} for _, row in slur_df.iterrows()],
            ids=[f"slur_{i}" for i in range(len(slur_df))]
        )
    
    if reclaimed_speech_collection.count() == 0:
        reclaimed_speech_collection.add(
            documents=reclaimed_df['text'].tolist(),
            metadatas=[{'source': row['source']} for _, row in reclaimed_df.iterrows()],
            ids=[f"reclaimed_{i}" for i in range(len(reclaimed_df))]
        )

load_initial_data()

# Neo4j Driver
neo4j_driver = None
if has_neo4j:
    try:
        from neo4j import GraphDatabase
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}")

# News Ingestion (Cache)
recent_news_cache = []
try:
    from GoogleNews import GoogleNews
    googlenews = GoogleNews(lang='en', period='7d')
    googlenews.search('immigration policy')
    all_articles = googlenews.result()
    recent_news_cache = [f"{a['title']}: {a['desc']}" for a in all_articles[:5]]
except Exception as e:
    print(f"News/KG Error: {e}")

# Pre-Retrieval Analysis
ENTITY_KEYWORDS = {
    "muslim", "muslims", "islam", "islamic", "hindu", "hindus", "christian", "christians",
    "jew", "jews", "jewish", "sikh", "sikhs", "buddhist", "buddhists", "atheist", "atheists",
    "asian", "asians", "african", "africans", "black", "blacks", "white", "whites",
    "latino", "latina", "hispanic", "indian", "indians", "chinese", "arab", "arabs",
    "liberal", "liberals", "conservative", "conservatives", "democrat", "democrats",
    "republican", "republicans", "leftist", "rightist", "communist", "fascist",
    "feminist", "feminists", "activist", "activists",
    "dalit", "dalits", "brahmin", "brahmins", "kshatriya", "vaishya", "shudra",
    "gay", "gays", "lesbian", "lesbians", "transgender", "trans", "queer", "lgbtq",
    "american", "americans", "british", "pakistani", "bangladeshi", "chinese",
    "immigrant", "immigrants", "refugee", "refugees", "migrant", "migrants"
}

CODED_TERMS = {
    "gloober", "zorp", "jogger", "dindu", "goy", "goyim", "globalist",
    "thug", "urban", "welfare queen", "illegal", "anchor baby",
    "cuck", "soyboy", "npc", "simp", "beta", "redpill", "bluepill",
    "chad", "incel", "mgtow", "thot", "roastie",
    "1488", "88", "14", "13/50", "despite",
    "dindu nuffin", "we wuz", "kangz", "sheeit", "ooga booga"
}

def pre_retrieval_analysis(text: str):
    doc = nlp(text)
    text_lower = text.lower()
    
    spacy_entities = [ent.text for ent in doc.ents]
    keyword_entities = [keyword.title() for keyword in ENTITY_KEYWORDS if keyword in text_lower]
    all_entities = list(set(spacy_entities + keyword_entities))
    
    oov_neologisms = [token.text for token in doc if token.is_oov and len(token.text) > 2]
    coded_found = [term for term in CODED_TERMS if term in text_lower]
    
    import re as regex
    number_codes = regex.findall(r'\b(1488|88|14|13\/50)\b', text)
    coded_found.extend(number_codes)
    
    all_neologisms = list(set(oov_neologisms + coded_found))
    indicator_matches = check_text_for_indicators(text)
    
    return {
        "entities": all_entities,
        "neologisms": all_neologisms,
        "indicator_matches": indicator_matches,
        "entity_count": len(all_entities),
        "neologism_count": len(all_neologisms)
    }

# RAG Agent
agent_executor = None
llm = None

if has_openrouter_api and slur_lexicon_db:
    try:
        from openai import OpenAI
        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        
        def call_llm(prompt: str) -> str:
            try:
                response = openrouter_client.chat.completions.create(
                    model="google/gemini-2.0-flash-exp:free",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM Error: {e}")
                return "Error calling LLM."
        
        llm = call_llm
        
        def rag_lexical_query(term: str) -> str:
            docs = slur_lexicon_db.similarity_search(term, k=1)
            if docs:
                return f"Lexicon Result for '{term}': {docs[0].page_content} (Target: {docs[0].metadata.get('target', 'N/A')})"
            return f"Lexicon Result for '{term}': Term not found."
        
        def rag_contextual_query(entity: str) -> str:
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
            
            if not events and recent_news_cache:
                events = [n for n in recent_news_cache if entity.lower() in n.lower()]
                
            if events:
                 return f"Contextual Result for '{entity}': Recently mentioned in news: " + " | ".join(events[:2])
            return f"Contextual Result for '{entity}': No recent news events found."
        
        def rag_counter_evidence_query(term: str) -> str:
            docs = reclaimed_speech_db.similarity_search(term, k=1)
            if docs:
                return f"Counter-Evidence Result for '{term}': Found a potential non-hateful usage: '{docs[0].page_content}'"
            return f"Counter-Evidence Result for '{term}': No instances found in reclaimed speech corpora."
        
        def execute_rag_queries(entities: list, neologisms: list) -> str:
            results = []
            for entity in entities[:3]:
                results.append(rag_contextual_query(entity))
            for term in neologisms[:3]:
                results.append(rag_lexical_query(term))
                results.append(rag_counter_evidence_query(term))
            if not results:
                results.append("No specific entities or neologisms found to query.")
            return " | ".join(results)
        
        agent_executor = execute_rag_queries
    except Exception as e:
        print(f"Could not initialize RAG agent: {e}")

# ML Model Training/Loading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rag_model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'models', 'vectorizer.pkl')
train_file_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv')

label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
tfidf_vectorizer = None
logistic_regression_model = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    logistic_regression_model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
elif os.path.exists(train_file_path):
    print("Training model (first run)...")
    df_train = pd.read_csv(train_file_path)
    text_col = 'comment_text' if 'comment_text' in df_train.columns else 'tweet'
    existing_labels = [col for col in label_columns if col in df_train.columns]
    
    if text_col in df_train.columns and existing_labels:
        for label in existing_labels:
            df_train[label] = (df_train[label] > 0).astype(int)
        
        df_train[text_col] = df_train[text_col].fillna('').astype(str).str.lower()
        df_train[text_col] = df_train[text_col].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
        df_train.dropna(subset=existing_labels, inplace=True)
        
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(df_train[text_col])
        
        logistic_regression_model = MultiOutputClassifier(
            LogisticRegression(solver='sag', class_weight='balanced', random_state=42, max_iter=2000)
        )
        logistic_regression_model.fit(X_train_tfidf, df_train[existing_labels])
        
        joblib.dump(logistic_regression_model, MODEL_PATH)
        joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
        print("Model trained and saved.")

# Helper Functions
def create_dynamic_prompt(original_sentence, context_from_rag, expert_type="General Hate"):
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
    doc = nlp(sentence.lower())
    for category, keywords in HATE_SPEECH_INDICATORS.items():
        if any(kw in doc.text for kw in keywords):
            return category
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP']:
            return "Political hate"
    return "General Hate"

def classify_hate_speech_combined(sentence: str) -> dict:
    classifications = {}
    if tfidf_vectorizer and logistic_regression_model:
        cleaned = sentence.lower()
        cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        sentence_tfidf = tfidf_vectorizer.transform([cleaned])
        pred_probs = logistic_regression_model.predict_proba(sentence_tfidf)
        probs_pos = np.array([p[:, 1] for p in pred_probs]).T[0]
        
        is_hateful = any((probs_pos >= 0.5).astype(int))
        classifications['model_result'] = "HATEFUL" if is_hateful else "NOT_HATEFUL"
        classifications['model_labels'] = [label_columns[i] for i, pred in enumerate((probs_pos >= 0.5).astype(int)) if pred == 1]
        classifications['probabilities'] = dict(zip(label_columns, [f"{p:.3f}" for p in probs_pos]))
    else:
        classifications['model_result'] = "Model not available"
    return classifications

# Main Analysis Function
def analyze_sentence(sentence: str):
    print(f"\nAnalyzing: '{sentence}'")
    
    # 1. Pre-Retrieval
    analysis = pre_retrieval_analysis(sentence)
    
    # 2. Type Detection
    expert_type = determine_sentence_type(sentence)
    
    # 3. ML Model
    model_result = classify_hate_speech_combined(sentence)
    
    # 4. RAG
    retrieved_context = {"output": "RAG agent not available"}
    if agent_executor:
        try:
            rag_output = agent_executor(analysis['entities'], analysis['neologisms'])
            retrieved_context = {"output": rag_output}
        except Exception as e:
            print(f"RAG Error: {e}")
    
    # 5. LLM Rationale
    llm_rationale = "LLM not available"
    if llm:
        try:
            prompt = create_dynamic_prompt(sentence, retrieved_context, expert_type)
            llm_rationale = llm(prompt)
        except Exception as e:
            print(f"LLM Error: {e}")
            
    # Output to console
    print(f"  Category: {expert_type}")
    print(f"  Model Result: {model_result['model_result']}")
    if model_result.get('model_labels'):
        print(f"  Labels: {model_result['model_labels']}")
    if str(retrieved_context['output']) != "No specific entities or neologisms found to query.":
         print(f"  RAG Context: {str(retrieved_context['output'])[:100]}...")

    return {
        "sentence": sentence,
        "category": expert_type,
        "model_result": model_result,
        "llm_rationale": llm_rationale
    }

if __name__ == "__main__":
    analyze_sentence("Test sentence")
