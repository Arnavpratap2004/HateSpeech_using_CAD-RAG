# Auto-extracted from cad_rag.ipynb

# === Cell 1 ===
# Block 1: Installation
!pip install langchain langchain-openai langchain-community neo4j psycopg2-binary pgvector spacy chromadb newsapi-python python-dotenv pandas langchain_google_genai openpyxl
!pip install 'langchain-community[docloaders]'
!python -m spacy download en_core_web_lg

# Block 2: Setup API Keys and Google Drive
import os
from google.colab import userdata, drive
from dotenv import load_dotenv

# Mount Google Drive to save our data
drive.mount('/content/drive')

# Load API keys from Colab secrets
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
os.environ['NEWS_API_KEY'] = userdata.get('NEWS_API_KEY')
os.environ['NEO4J_URI'] = userdata.get('NEO4J_URI')
os.environ['NEO4J_USERNAME'] = userdata.get('NEO4J_USERNAME')
os.environ['NEO4J_PASSWORD'] = userdata.get('NEO4J_PASSWORD')

print("Environment setup complete. API keys and Google Drive are configured.")

# Block 3: Initialize ChromaDB Vector Store
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Define the path in your Google Drive for persistence
CHROMA_PATH = "/content/drive/MyDrive/cad_rag_chroma"

# Initialize the embedding function
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get the collections for our lexicons
slur_lexicon_collection = chroma_client.get_or_create_collection(
    name="slur_lexicon",
    metadata={"hnsw:space": "cosine"}
)
reclaimed_speech_collection = chroma_client.get_or_create_collection(
    name="reclaimed_speech",
    metadata={"hnsw:space": "cosine"}
)

# Create LangChain vector store objects
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

print(f"ChromaDB initialized at: {CHROMA_PATH}")
print(f"Number of items in slur lexicon: {slur_lexicon_db._collection.count()}")
print(f"Number of items in reclaimed speech corpus: {reclaimed_speech_db._collection.count()}")

# Block 4: Populate Knowledge Base with Initial Data
import pandas as pd
from neo4j import GraphDatabase
import spacy
from newsapi import NewsApiClient
import os

# --- 1. Populate Slur Lexicon and Reclaimed Speech Corpus ---
# Manually create some initial data
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
    'text':['', ''],
    'source': ['Community Forum A', 'Academic Paper on Linguistics']
}
reclaimed_df = pd.DataFrame(reclaimed_data)

# Add to ChromaDB (if not already present)
if slur_lexicon_db._collection.count() == 0:
    slur_lexicon_db.add_texts(
        texts=slur_df['definition'].tolist(),
        metadatas=[{'term': row['term'], 'target': row['target_group']} for _, row in slur_df.iterrows()],
        ids=[f"slur_{i}" for i in range(len(slur_df))]
    )
    print("Added initial data to slur lexicon.")

if reclaimed_speech_db._collection.count() == 0:
    reclaimed_speech_db.add_texts(
        texts=reclaimed_df['text'].tolist(),
        metadatas=[{'source': row['source']} for _, row in reclaimed_df.iterrows()],
        ids=[f"reclaimed_{i}" for i in range(len(reclaimed_df))]
    )
    print("Added initial data to reclaimed speech corpus.")


# --- 2. Simulate Real-Time News Ingestion for Knowledge Graph ---
# Initialize clients
nlp = spacy.load("en_core_web_lg")
newsapi = NewsApiClient(api_key=os.environ.get('NEWS_API_KEY'))
neo4j_driver = GraphDatabase.driver(os.environ.get('NEO4J_URI'), auth=(os.environ.get('NEO4J_USERNAME'), os.environ.get('NEO4J_PASSWORD')))

def ingest_article_to_kg(tx, article):
    """A function to ingest a single news article into the Neo4j Knowledge Graph."""
    tx.run("""
        MERGE (e:Event {url: $url})
        ON CREATE SET e.title = $title, e.summary = $summary, e.date = datetime($date)
        """,
        url=article['url'],
        title=article['title'],
        summary=article['description'] or "No description available.",
        date=article['publishedAt']
    )
    doc = nlp(article['title'])
    relevant_entity_types = ['PERSON', 'ORG', 'GPE', 'NORP']
    for ent in doc.ents:
        if ent.label_ in relevant_entity_types:
            tx.run("""
                MERGE (e:Event {url: $url})
                MERGE (en:Entity {name: $name})
                ON CREATE SET en.type = $type
                MERGE (e)-->(en)
                """,
                url=article['url'], name=ent.text, type=ent.label_
            )

print("\nFetching and ingesting recent news articles into Knowledge Graph...")
try:
    all_articles = newsapi.get_everything(q='immigration policy', language='en', sort_by='publishedAt', page_size=10)
    with neo4j_driver.session() as session:
        for article in all_articles['articles']:
            session.execute_write(ingest_article_to_kg, article)
    print(f"Successfully ingested {len(all_articles['articles'])} articles into Neo4j.")
except Exception as e:
    print(f"Could not fetch news articles. Error: {e}. This may be due to API rate limits or incorrect Neo4j credentials.")

# Block 5: Pre-Retrieval Analysis Function
def pre_retrieval_analysis(text: str):
    """Extracts named entities and potential neologisms from text."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    neologisms = [token.text for token in doc if token.is_oov]
    return {"entities": list(set(entities)), "neologisms": list(set(neologisms))}

# Block 6: RAG Core - Multi-Query Agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase
import os

@tool
def lexical_query(term: str) -> str:
    """Searches the Evolving Slur Lexicon for a definition and origin of a given term."""
    print(f"--- EXECUTING LEXICAL QUERY for '{term}' ---")
    docs = slur_lexicon_db.similarity_search(term, k=1)
    if docs:
        return f"Lexicon Result for '{term}': {docs[0].page_content} (Target: {docs[0].metadata.get('target', 'N/A')})"
    return f"Lexicon Result for '{term}': Term not found."

# neo4j_driver is already initialized in Block 4
@tool
def contextual_query(entity: str) -> str:
    """Searches the Knowledge Graph for recent news or controversies related to a specific entity."""
    print(f"--- EXECUTING CONTEXTUAL QUERY for '{entity}' ---")
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (e:Event)-->(en:Entity)
            WHERE toLower(en.name) CONTAINS toLower($entity)
            RETURN e.title, e.summary
            ORDER BY e.date DESC
            LIMIT 2
            """, entity=entity)
        events = [f"{record['e.title']}: {record['e.summary']}" for record in result]
    if events:
        return f"Contextual Result for '{entity}': Recently mentioned in news: " + " | ".join(events)
    return f"Contextual Result for '{entity}': No recent news events found in the knowledge graph."

@tool
def counter_evidence_query(term: str) -> str:
    """Searches the Corpus of Reclaimed Speech for non-hateful uses of a given term."""
    print(f"--- EXECUTING COUNTER-EVIDENCE QUERY for '{term}' ---")
    docs = reclaimed_speech_db.similarity_search(term, k=1)
    if docs:
        return f"Counter-Evidence Result for '{term}': Found a potential non-hateful usage: '{docs[0].page_content}'"
    return f"Counter-Evidence Result for '{term}': No instances found in reclaimed speech corpora."

tools = [lexical_query, contextual_query, counter_evidence_query]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ.get("GOOGLE_API_KEY"))

agent_prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Input: {input}
{agent_scratchpad}
"""
agent_prompt = PromptTemplate.from_template(agent_prompt_template)

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Start of new model integration and refined functions ---

# Import necessary libraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score, classification_report
import re
import pandas as pd
import os
import numpy as np

# --- Load and Train Original Model (jigsaw-toxic-comment-classification-challenge) ---
# Define the path to the train.csv file
train_file_path_original = '/tmp/train.csv'
test_file_path_original = '/tmp/traintest.csv' # Path to the test data

# Define the label columns used during training
label_columns = ['hate_speech', 'offensive_language', 'neither']


# Check if train.csv exists.
if os.path.exists(train_file_path_original):
    try:
        # Load the train.csv file into a pandas DataFrame
        df_train = pd.read_csv(train_file_path_original)
        print("\ntrain.csv loaded successfully for original model training.")

        # **MERGED CODE**: Binarize the label columns.
        print("\nBinarizing label columns for training data...")
        for label in label_columns:
            df_train[label] = (df_train[label] > 0).astype(int)
        print("Binarization complete.")

        # --- Analyze Class Distribution (Train) ---
        print("\nClass distribution in training data (after binarization):")
        print(df_train[label_columns].sum())

        # --- Remove rows with NaN labels ---
        initial_rows = len(df_train)
        df_train.dropna(subset=label_columns, inplace=True)
        rows_removed = initial_rows - len(df_train)
        print(f"Removed {rows_removed} rows with NaN labels from train.csv.")


        # Perform basic text cleaning for original model
        print("\nStarting text cleaning and preprocessing steps for original model training...")
        df_train['tweet'] = df_train['tweet'].fillna('')
        df_train['tweet'] = df_train['tweet'].str.lower()
        df_train['tweet'] = df_train['tweet'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
        df_train['tweet'] = df_train['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        print("Text cleaning and preprocessing steps completed for original model training.")

        # Define the features (text) and labels for original model
        X_original = df_train['tweet']
        y_original = df_train[label_columns]

        # MODIFICATION: Increased max_features to 10000
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)

        # Fit the vectorizer and transform data for original model
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_original)

        # Initialize and train the original LogisticRegression model with class_weight='balanced'
        logistic_regression_model = MultiOutputClassifier(LogisticRegression(solver='sag', class_weight='balanced', random_state=42))
        print("\nOriginal Logistic Regression model initialized with class_weight='balanced'.")

        print("\nTraining the original Logistic Regression model...")
        logistic_regression_model.fit(X_train_tfidf, y_original)
        print("Original model training completed.")

        # --- Evaluate Original Model ---
        if os.path.exists(test_file_path_original):
            try:
                df_test = pd.read_csv(test_file_path_original) # Read as CSV
                print(f"\n'{test_file_path_original}' loaded successfully for original model evaluation.")

                # **MERGED CODE**: Binarize the label columns for the test data as well.
                print("\nBinarizing label columns for test data...")
                for label in label_columns:
                    df_test[label] = (df_test[label] > 0).astype(int)
                print("Binarization complete.")

                # --- Analyze Class Distribution (Test) ---
                print("\nClass distribution in test data (after binarization):")
                print(df_test[label_columns].sum())

                # Assuming the test set has the same structure and label columns as the training set
                # Convert 'comment_text' to string type to handle potential non-string data
                df_test['tweet'] = df_test['tweet'].astype(str).fillna('')
                X_test_original = df_test['tweet']
                X_test_original = X_test_original.str.lower()
                X_test_original = X_test_original.apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
                X_test_original = X_test_original.apply(lambda x: re.sub(r'\s+', ' ', x).strip())

                y_test_original = df_test[label_columns].dropna()
                X_test_original = X_test_original[y_test_original.index]


                # Transform the test data using the fitted vectorizer
                X_test_tfidf = tfidf_vectorizer.transform(X_test_original)

                # Make predictions on the test data using predict_proba
                y_pred_proba_original = logistic_regression_model.predict_proba(X_test_tfidf)

                # Apply a custom threshold to get final predictions
                # Note: predict_proba returns a list of arrays, one for each label.
                # We stack them and then apply the threshold.
                probs_positive_class = np.array([p[:, 1] for p in y_pred_proba_original]).T

                # MODIFICATION: Increased threshold to 0.8 for higher precision
                custom_threshold = 0.8
                y_pred_original = (probs_positive_class >= custom_threshold).astype(int)
                print(f"\nUsing custom prediction threshold for high precision: {custom_threshold}")


                # Calculate evaluation metrics
                print("\nOriginal Model Evaluation Metrics:")
                for i, label in enumerate(label_columns):
                    print(f"\nMetrics for label: {label}")
                    print(f"  Accuracy: {accuracy_score(y_test_original.iloc[:, i], y_pred_original[:, i]):.4f}")
                    print(f"  Precision: {precision_score(y_test_original.iloc[:, i], y_pred_original[:, i], zero_division=0):.4f}")
                    print(f"  Recall: {recall_score(y_test_original.iloc[:, i], y_pred_original[:, i], zero_division=0):.4f}")
                    print(f"  F1 Score: {f1_score(y_test_original.iloc[:, i], y_pred_original[:, i], zero_division=0):.4f}")

                # Calculate overall metrics
                print("\nOverall Original Model Metrics:")
                print(f"  Hamming Loss: {hamming_loss(y_test_original, y_pred_original):.4f}")
                print(f"  Jaccard Score (samples): {jaccard_score(y_test_original, y_pred_original, average='samples'):.4f}")


            except Exception as e:
                print(f"An error occurred during original model evaluation: {e}")
        else:
            print(f"\nError: '{test_file_path_original}' not found for original model evaluation. Please ensure the dataset is in the /tmp/ directory.")

    except Exception as e:
        print(f"An error occurred during the original model training process: {e}")
else:
    print(f"\nError: {train_file_path_original} not found for original model training. Please ensure the dataset is in the /tmp/ directory.")



# --- Refined Sentence Type Analysis and Dynamic Prompting ---

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
    """
    Analyses a sentence to determine its potential hate speech category based on keywords, entities, and patterns.

    Args:
        sentence: The input sentence.

    Returns:
        A string indicating the most likely hate speech category.
    """
    doc = nlp(sentence.lower())

    religion_indicators = ["muslims are", "islamic extremists", "hindu nationalists", "christian fundamentalists", "jewish conspiracy", "sikh separatists", "buddhist persecution", "anti-muslim", "anti-christian", "anti-jewish", "anti-sikh", "anti-buddhist", "religious fanatic", "infidels", "blasphemy", "jihad", "crusade", "sharia", "temple demolition", "mosque attack", "church bombing", "synagogue vandalism", "religious fanatics", "people of faith are"]
    political_indicators = ["opposition are traitors", "government is corrupt", "political elite is", "leftist agenda is", "right-wing extremist are", "political enemies are", "election fraud is", "deep state is", "political propaganda is", "political violence is", "political purges are", "congress party is thug"]
    gender_indicators = ["women belong in the kitchen", "men are superior to", "feminazis are", "male fragility is", "gender fluid is not real", "transgenders are not", "sexual predators are", "misogynistic comments", "patriarchal oppression is", "sexist remarks are", "women are too emotional"]
    caste_indicators = ["lower castes are inferior", "upper castes are", "dalit oppression is", "brahminical supremacy is", "caste hierarchy is", "caste-based violence is"]
    cyberbullying_indicators = ["going to dox that person", "expose on social media", "online harassment campaign", "troll army is", "cyberstalking is", "sending threats online", "online abuse is", "internet mob is", "comment warriors are"]
    subtle_hate_indicators = ["you know the type of people", "those kind of people always", "some groups just always", "they're all the same", "it's just a joke relax", "can't you take a joke", "playing the victim card again", "coded language used", "dog whistle politics", "sarcastic remark about"]

    if any(phrase in doc.text for phrase in caste_indicators):
        return "Caste-based hate"
    if any(phrase in doc.text for phrase in religion_indicators):
        return "Religion-based hate"
    if any(phrase in doc.text for phrase in gender_indicators):
        return "Gender-based hate"
    if any(phrase in doc.text for phrase in political_indicators):
        return "Political hate"
    if any(phrase in doc.text for phrase in cyberbullying_indicators):
        return "Cyberbullying & personal attacks"
    if any(phrase in doc.text for phrase in subtle_hate_indicators):
        return "Subtle hate (sarcasm, euphemism, coded language)"

    religion_keywords = ["muslim", "islam", "hindu", "christian", "jew", "sikh", "buddhist", "religion", "religious", "faith", "allah", "jesus", "temple", "mosque", "church"]
    political_keywords = ["political", "government", "party", "vote", "election", "politician", "liberal", "conservative", "democrat", "republican", "traitor", "propaganda", "regime", "congress"]
    gender_keywords = ["man", "woman", "male", "female", "guy", "girl", "boy", "she", "he", "her", "him", "gender", "feminist", "sexist", "women", "men", "girls", "boys"]
    caste_keywords = ["caste", "dalit", "brahmin", "shudra", "kshatriya", "varna", "jati", "chamar"]
    cyberbullying_keywords = ["online", "internet", "social media", "forum", "comment", "twitter", "facebook", "instagram", "tiktok", "snapchat", "dox", "troll", "cyber"]
    subtle_hate_keywords = ["subtle", "implicit", "coded", "sarcasm", "euphemism", "innuendo", "different", "type", "sort"]

    if any(keyword in doc.text for keyword in caste_keywords):
        return "Caste-based hate"
    if any(keyword in doc.text for keyword in religion_keywords):
        return "Religion-based hate"
    if any(keyword in doc.text for keyword in gender_keywords):
        return "Gender-based hate"
    if any(keyword in doc.text for keyword in political_keywords):
        return "Political hate"
    if any(keyword in doc.text for keyword in cyberbullying_keywords):
        return "Cyberbullying & personal attacks"
    if any(keyword in doc.text for keyword in subtle_hate_keywords):
        return "Subtle hate (sarcasm, euphemism, coded language)"

    if not any(indicator in doc.text for indicators_list in [religion_indicators, political_indicators, gender_indicators, caste_indicators, cyberbullying_indicators, subtle_hate_indicators] for indicator in indicators_list) and \
       not any(keyword in doc.text for keywords_list in [religion_keywords, political_keywords, gender_keywords, caste_keywords, cyberbullying_keywords, subtle_hate_keywords] for keyword in keywords_list):
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP']:
                return "Political hate"

    return "General Hate"

# --- Combined Classification Function ---
def classify_hate_speech_combined(sentence: str) -> dict:
    """
    Classifies a sentence using the trained model.

    Args:
        sentence: The input sentence to classify.

    Returns:
        A dictionary containing model classifications.
    """
    classifications = {}

    if 'tfidf_vectorizer' in globals() and 'logistic_regression_model' in globals():
        try:
            cleaned_sentence = sentence.lower()
            cleaned_sentence = re.sub(r'[^a-z0-9\s]', '', cleaned_sentence)
            cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
            sentence_tfidf = tfidf_vectorizer.transform([cleaned_sentence])

            # Predict probabilities and apply the high-precision threshold
            pred_probs = logistic_regression_model.predict_proba(sentence_tfidf)
            probs_positive_class = np.array([p[:, 1] for p in pred_probs]).T[0]

            # Use the same high threshold as in evaluation
            custom_threshold = 0.8
            predictions = (probs_positive_class >= custom_threshold).astype(int)

            is_hateful_original = any(predictions)
            classifications['original_model'] = "HATEFUL" if is_hateful_original else "NOT_HATEFUL"
            predicted_labels = [label_columns[i] for i, pred in enumerate(predictions) if pred == 1]
            classifications['original_model_labels'] = predicted_labels

        except Exception as e:
            classifications['original_model'] = f"Error: {e}"
    else:
        classifications['original_model'] = "Model not available."


    return classifications

# --- Main Analysis Function (analyze_sentence) ---
def analyze_sentence(sentence: str):
    """
    Analyses a sentence for potential hate speech using the defined RAG model
    and the trained classification models.

    Args:
        sentence: The input sentence to analyze.

    Returns:
        A dictionary containing model classifications and the LLM rationale.
    """
    print(f"--- Analyzing Sentence: '{sentence}' ---")

    # 1. Pre-Retrieval Analysis
    analysis_result = pre_retrieval_analysis(sentence)
    print(f"Analysis Result: {analysis_result}")

    # 2. Determine Sentence Type for Expert Routing (for RAG)
    expert_type = determine_sentence_type(sentence)
    print(f"Determined Expert Type (for RAG): {expert_type}")

    # 3. Trained Model Classifications
    model_classifications = classify_hate_speech_combined(sentence)
    print(f"\nTrained Model Classifications: {model_classifications}")

    # 4. RAG Core - Multi-Query Agent
    analysis_input_str = f"Analyzed text contains entities: {analysis_result.get('entities', [])} and neologisms: {analysis_result.get('neologisms', [])}"
    print(f"Agent Input: {analysis_input_str}")
    retrieved_context = {"output": "Agent execution failed."} # Initialize retrieved_context
    try:
        retrieved_context = agent_executor.invoke({"input": analysis_input_str})
        print(f"\nRetrieved Context: {retrieved_context['output']}")
    except NameError:
        print("Error: agent_executor is not defined. Please ensure Block 6 has been executed.")
    except Exception as e:
        print(f"An error occurred during Agent execution: {e}")


    # 5. Dynamic Prompting and Final Decision (LLM Rationale)
    final_prompt_for_expert = create_dynamic_prompt(
        original_sentence=sentence,
        context_from_rag=retrieved_context,
        expert_type=expert_type
    )
    print("\n--- DYNAMIC PROMPT FOR EXPERT (for Rationale) ---")
    print(final_prompt_for_expert)

    try:
        expert_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ.get("GOOGLE_API_KEY"))
        final_decision_llm = expert_llm.invoke(final_prompt_for_expert)
        llm_rationale = final_decision_llm.content
        print("\n--- LLM RATIONALE ---")
        print(llm_rationale)
    except Exception as e:
        print(f"An error occurred during LLM rationale generation: {e}")
        llm_rationale = f"LLM rationale generation failed: {e}"

    return {"model_classifications": model_classifications, "llm_rationale": llm_rationale}


# --- Example Usage ---
test_sentence = "All muslims should be kicked out"
analysis_output = analyze_sentence(test_sentence)
print("\n--- Final Analysis Output (Model + LLM) ---")
print(analysis_output)

