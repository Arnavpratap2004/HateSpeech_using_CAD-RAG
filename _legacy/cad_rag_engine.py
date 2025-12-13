import os
import re
import warnings
import pandas as pd
import numpy as np
import spacy
import joblib
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from GoogleNews import GoogleNews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')

class CADRAGEngine:
    def __init__(self, script_dir=None):
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self.env_path = os.path.join(self.script_dir, '.env')
        self.chroma_path = os.path.join(self.script_dir, "cad_rag_chroma")
        self.model_path = os.path.join(self.script_dir, 'cad_rag_model.pkl')
        self.vectorizer_path = os.path.join(self.script_dir, 'cad_rag_vectorizer.pkl')
        self.train_file_path = os.path.join(self.script_dir, 'train.csv')
        self.test_file_path = os.path.join(self.script_dir, 'traintest.csv')
        
        # State
        self.nlp = None
        self.chroma_client = None
        self.slur_lexicon_db = None
        self.reclaimed_speech_db = None
        self.tfidf_vectorizer = None
        self.logistic_regression_model = None
        self.openrouter_client = None
        self.neo4j_driver = None
        self.recent_news_cache = []
        self.initialized = False

        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def initialize(self):
        print("\n[Engine] Initializing CAD-RAG Engine...")
        
        # 1. Environment
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path)
            print("  - Loaded .env")
        
        self.openrouter_key = os.environ.get('OPENROUTER_API_KEY', '')
        self.neo4j_uri = os.environ.get('NEO4J_URI', '')
        self.neo4j_user = os.environ.get('NEO4J_USERNAME', '')
        self.neo4j_pass = os.environ.get('NEO4J_PASSWORD', '')

        # 2. SpaCy
        print("  - Loading SpaCy...")
        self.nlp = spacy.load("en_core_web_lg")

        # 3. Vector Store
        print("  - Loading ChromaDB & Embeddings...")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.slur_lexicon_db = Chroma(
            client=self.chroma_client,
            collection_name="slur_lexicon",
            embedding_function=embedding_function,
        )
        self.reclaimed_speech_db = Chroma(
            client=self.chroma_client,
            collection_name="reclaimed_speech",
            embedding_function=embedding_function,
        )

        # 4. News
        print("  - Fetching News (GoogleNews)...")
        try:
            googlenews = GoogleNews(lang='en', period='7d')
            googlenews.search('immigration policy')
            results = googlenews.result()
            self.recent_news_cache = [f"{a['title']}: {a['desc']}" for a in results[:5]]
            print(f"    Fetched {len(self.recent_news_cache)} articles")
        except Exception as e:
            print(f"    News fetch failed: {e}")

        # 5. Neo4j (Disabled/Removed)
        self.neo4j_driver = None
        # if self.neo4j_uri and self.neo4j_user and self.neo4j_pass:
        #     try:
        #         from neo4j import GraphDatabase
        #         driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_pass))
        #         # Quick check
        #         with driver.session() as session:
        #             session.run("RETURN 1").single()
        #         self.neo4j_driver = driver
        #         print("  - Neo4j Connected")
        #     except Exception:
        #         print("  - Neo4j Connection Failed (Skipping)")

        # 6. OpenRouter
        if self.openrouter_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
            )
            print("  - OpenRouter Client Ready")

        # 7. ML Model
        print("  - Loading ML Model...")
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.logistic_regression_model = joblib.load(self.model_path)
            self.tfidf_vectorizer = joblib.load(self.vectorizer_path)
            print("    Loaded saved model.")
        else:
            print("    Saved model not found. Training now...")
            self._train_model()

        self.initialized = True
        print("[Engine] Initialization Complete!\n")

    def _train_model(self):
        if not os.path.exists(self.train_file_path):
             raise FileNotFoundError(f"Training data not found at {self.train_file_path}")
        
        df_train = pd.read_csv(self.train_file_path)
        
        text_col = 'comment_text' if 'comment_text' in df_train.columns else 'tweet'
        existing_labels = [col for col in self.label_columns if col in df_train.columns]

        if not (text_col in df_train.columns and existing_labels):
             raise ValueError("Training data missing required columns")

        # Preprocess
        for label in existing_labels:
            df_train[label] = (df_train[label] > 0).astype(int)
        
        df_train[text_col] = df_train[text_col].fillna('').astype(str).str.lower()
        df_train[text_col] = df_train[text_col].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
        df_train[text_col] = df_train[text_col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        df_train.dropna(subset=existing_labels, inplace=True)

        X_train = df_train[text_col]
        y_train = df_train[existing_labels]

        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)

        # Train
        self.logistic_regression_model = MultiOutputClassifier(
            LogisticRegression(solver='sag', class_weight='balanced', random_state=42, max_iter=2000)
        )
        self.logistic_regression_model.fit(X_train_tfidf, y_train)
        
        # Save
        joblib.dump(self.logistic_regression_model, self.model_path)
        joblib.dump(self.tfidf_vectorizer, self.vectorizer_path)
        print("    Model trained and saved.")

    # --- Analysis Functions ---

    def analyze_sentence(self, sentence: str):
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # 1. Pre-Retrieval
        doc = self.nlp(sentence)
        entities = list(set([ent.text for ent in doc.ents]))
        neologisms = list(set([token.text for token in doc if token.is_oov]))
        category = self._determine_category(doc)

        # 2. ML Prediction
        model_res = self._predict_ml(sentence)

        # 3. RAG Retrieval
        rag_context = self._run_rag(entities, neologisms)

        # 4. LLM Analysis
        llm_rationale = self._call_llm(sentence, rag_context, category)

        return {
            "sentence": sentence,
            "category": category,
            "entities": entities,
            "neologisms": neologisms,
            "model_result": model_res,
            "rag_context": rag_context,
            "llm_rationale": llm_rationale
        }

    def _determine_category(self, doc):
        text = doc.text.lower()
        indicators = {
            "Caste-based": ["lower castes", "upper castes", "dalit", "brahmin", "caste"],
            "Religion-based": ["muslim", "islam", "hindu", "christian", "religion"],
            "Gender-based": ["women", "men", "feminist", "sexist"],
            "Political": ["political", "government", "traitor", "election"],
            "Cyberbullying": ["die", "stupid", "idiot", "hate", "ugly"],
        }
        for cat, kws in indicators.items():
            if any(kw in text for kw in kws): return cat
        return "General Hate"

    def _predict_ml(self, sentence):
        if not self.logistic_regression_model: return {"result": "Model unavailable"}
        
        cleaned = re.sub(r'[^a-z0-9\s]', '', sentence.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        vec = self.tfidf_vectorizer.transform([cleaned])
        probs = self.logistic_regression_model.predict_proba(vec)
        
        # Get positive class probs
        probs_pos = np.array([p[:, 1] for p in probs]).T[0] 
        threshold = 0.5
        preds = (probs_pos >= threshold).astype(int)
        
        result = "HATEFUL" if any(preds) else "NOT_HATEFUL"
        labels = [self.label_columns[i] for i, p in enumerate(preds) if p == 1]
        
        return {"result": result, "labels": labels, "probs": dict(zip(self.label_columns, probs_pos))}

    def _run_rag(self, entities, neologisms):
        results = []
        
        # Contextual (News/Neo4j)
        for ent in entities[:2]:
            found = False
            # Check Neo4j
            if self.neo4j_driver:
                try:
                    with self.neo4j_driver.session() as session:
                        res = session.run("MATCH (e:Event)-->(en:Entity) WHERE toLower(en.name) CONTAINS toLower($E) RETURN e.title, e.summary LIMIT 2", E=ent)
                        recs = [f"{r['e.title']}" for r in res]
                        if recs: results.append(f"News ({ent}): " + "; ".join(recs)); found = True
                except: pass
            
            # Fallback to cache
            if not found and self.recent_news_cache:
                hits = [n for n in self.recent_news_cache if ent.lower() in n.lower()]
                if hits: results.append(f"News ({ent}): " + "; ".join(hits[:1]))

        # Lexical (Chroma)
        for term in neologisms[:2]:
            docs = self.slur_lexicon_db.similarity_search(term, k=1)
            if docs: results.append(f"Lexicon ({term}): {docs[0].page_content}")
            
            docs2 = self.reclaimed_speech_db.similarity_search(term, k=1)
            if docs2: results.append(f"Reclaimed ({term}): {docs2[0].page_content}")
            
        return " | ".join(results) if results else "No specific context found."

    def _call_llm(self, sentence, context, category):
        if not self.openrouter_client: return "LLM not configured"
        
        prompt = f"""You are an expert at detecting {category} speech. Analyze this sentence.
Sentence: '{sentence}'
Context: {context}
Task: Classify as HATEFUL/NOT_HATEFUL and explain why based on the sentence and context."""

        try:
            print(f"    (Calling OpenRouter: nousresearch/hermes-3-llama-3.1-405b:free)...")
            response = self.openrouter_client.chat.completions.create(
                model="nousresearch/hermes-3-llama-3.1-405b:free",
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "HTTP-Referer": "https://localhost:3000",
                    "X-Title": "CAD-RAG Local"
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {str(e)[:100]}..."
