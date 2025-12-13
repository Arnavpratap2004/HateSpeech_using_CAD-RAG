# CAD-RAG Local Test Script
# Adapted from cad_rag.ipynb for local execution
# This script tests the ML model with the provided CSV files

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(SCRIPT_DIR, 'train.csv')
test_file_path = os.path.join(SCRIPT_DIR, 'traintest.csv')

print("=" * 60)
print("CAD-RAG Local Test - ML Model Training and Evaluation")
print("=" * 60)

# --- Load Training Data (Jigsaw Dataset: train.csv) ---
print(f"\nLoading training data from: {train_file_path}")
if not os.path.exists(train_file_path):
    print(f"ERROR: {train_file_path} not found!")
    exit(1)

df_train = pd.read_csv(train_file_path)
print(f"Training data loaded successfully. Shape: {df_train.shape}")
print(f"Columns: {list(df_train.columns)}")

# Jigsaw dataset columns
jigsaw_label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
jigsaw_text_column = 'comment_text'

# Verify columns exist
if jigsaw_text_column not in df_train.columns:
    print(f"ERROR: '{jigsaw_text_column}' column not found in training data!")
    exit(1)

existing_jigsaw_labels = [col for col in jigsaw_label_columns if col in df_train.columns]
print(f"Label columns found: {existing_jigsaw_labels}")

# --- Binarize Labels (already binary in Jigsaw dataset) ---
print("\nVerifying binary labels...")
for label in existing_jigsaw_labels:
    df_train[label] = (df_train[label] > 0).astype(int)
print("Labels verified as binary.")

# --- Class Distribution ---
print(f"\nClass distribution in training data:")
print(df_train[existing_jigsaw_labels].sum())

# --- Remove NaN rows ---
initial_rows = len(df_train)
df_train.dropna(subset=existing_jigsaw_labels + [jigsaw_text_column], inplace=True)
rows_removed = initial_rows - len(df_train)
print(f"Removed {rows_removed} rows with NaN values.")

# --- Text Cleaning ---
print("\nCleaning text data...")
df_train[jigsaw_text_column] = df_train[jigsaw_text_column].fillna('').astype(str)
df_train[jigsaw_text_column] = df_train[jigsaw_text_column].str.lower()
df_train[jigsaw_text_column] = df_train[jigsaw_text_column].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
df_train[jigsaw_text_column] = df_train[jigsaw_text_column].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
print("Text cleaning completed.")

# --- Prepare Features and Labels ---
X_train = df_train[jigsaw_text_column]
y_train = df_train[existing_jigsaw_labels]

print(f"\nTraining data prepared:")
print(f"  - Number of samples: {len(X_train)}")
print(f"  - Number of labels: {len(existing_jigsaw_labels)}")

# --- TF-IDF Vectorization ---
print("\nInitializing TF-IDF Vectorizer (max_features=10000)...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# --- Train Model ---
print("\nTraining Logistic Regression model with class_weight='balanced'...")
logistic_regression_model = MultiOutputClassifier(
    LogisticRegression(solver='sag', class_weight='balanced', random_state=42, max_iter=1000)
)
logistic_regression_model.fit(X_train_tfidf, y_train)
print("Model training completed!")

# --- Load Test Data (Davidson Dataset: traintest.csv) ---
print("\n" + "=" * 60)
print("Loading Test Data (traintest.csv)")
print("=" * 60)

if not os.path.exists(test_file_path):
    print(f"ERROR: {test_file_path} not found!")
    exit(1)

df_test = pd.read_csv(test_file_path)
print(f"Test data loaded successfully. Shape: {df_test.shape}")
print(f"Columns: {list(df_test.columns)}")

# Davidson dataset columns
davidson_label_columns = ['hate_speech', 'offensive_language', 'neither']
davidson_text_column = 'tweet'

# Verify columns
if davidson_text_column not in df_test.columns:
    print(f"ERROR: '{davidson_text_column}' column not found in test data!")
    exit(1)

existing_davidson_labels = [col for col in davidson_label_columns if col in df_test.columns]
print(f"Label columns found: {existing_davidson_labels}")

# --- Binarize Labels ---
print("\nBinarizing label columns...")
for label in existing_davidson_labels:
    df_test[label] = (df_test[label] > 0).astype(int)
print("Labels binarized.")

# --- Class Distribution ---
print(f"\nClass distribution in test data:")
print(df_test[existing_davidson_labels].sum())

# --- Clean Test Text ---
print("\nCleaning test text data...")
df_test[davidson_text_column] = df_test[davidson_text_column].astype(str).fillna('')
X_test = df_test[davidson_text_column].str.lower()
X_test = X_test.apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
X_test = X_test.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
print("Text cleaning completed.")

# --- Prepare test labels ---
y_test = df_test[existing_davidson_labels].dropna()
X_test = X_test[y_test.index]

print(f"\nTest data prepared:")
print(f"  - Number of samples: {len(X_test)}")

# --- Transform Test Data ---
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- Note about Dataset Mismatch ---
print("\n" + "=" * 60)
print("IMPORTANT: DATASET MISMATCH DETECTED")
print("=" * 60)
print("Training data (train.csv): Jigsaw Toxic Comment dataset")
print("  Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate")
print("\nTest data (traintest.csv): Davidson Hate Speech dataset")
print("  Labels: hate_speech, offensive_language, neither")
print("\nThese datasets have different label schemas!")
print("We will demonstrate the model's predictions on test data,")
print("but metrics may not be meaningful due to label mismatch.")

# --- Make Predictions ---
print("\n" + "=" * 60)
print("MODEL PREDICTIONS ON TEST DATA")
print("=" * 60)

print("\nMaking predictions using trained model...")
y_pred_proba = logistic_regression_model.predict_proba(X_test_tfidf)

# Apply custom threshold
probs_positive_class = np.array([p[:, 1] for p in y_pred_proba]).T
custom_threshold = 0.5
y_pred = (probs_positive_class >= custom_threshold).astype(int)
print(f"Using prediction threshold: {custom_threshold}")

# --- Show Predictions Distribution ---
print("\nPrediction distribution (Jigsaw labels on Davidson data):")
for i, label in enumerate(existing_jigsaw_labels):
    positive_preds = y_pred[:, i].sum()
    print(f"  {label}: {positive_preds} positive predictions ({100*positive_preds/len(y_pred):.1f}%)")

# --- Test with Example Sentences ---
print("\n" + "=" * 60)
print("TESTING WITH EXAMPLE SENTENCES")
print("=" * 60)

test_sentences = [
    "All muslims should be kicked out",
    "Have a great day everyone!",
    "These people are disgusting and should be removed",
    "I love this beautiful weather",
    "Women belong in the kitchen",
    "You are such an idiot, I hate you",
    "Thank you for your help, you're amazing!",
]

print("\nClassifying example sentences:")
for sentence in test_sentences:
    # Clean the sentence
    cleaned = sentence.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Vectorize
    sentence_tfidf = tfidf_vectorizer.transform([cleaned])
    
    # Predict
    pred_probs = logistic_regression_model.predict_proba(sentence_tfidf)
    probs_pos = np.array([p[:, 1] for p in pred_probs]).T[0]
    predictions = (probs_pos >= 0.5).astype(int)
    
    is_toxic = any(predictions)
    predicted_labels = [existing_jigsaw_labels[i] for i, pred in enumerate(predictions) if pred == 1]
    
    print(f"\n  Sentence: \"{sentence}\"")
    print(f"  Classification: {'TOXIC/HATEFUL' if is_toxic else 'NOT TOXIC/HATEFUL'}")
    if predicted_labels:
        print(f"  Labels: {predicted_labels}")
    # Show top probabilities
    sorted_probs = sorted(zip(existing_jigsaw_labels, probs_pos), key=lambda x: x[1], reverse=True)
    top_probs = sorted_probs[:3]
    print(f"  Top probabilities: {', '.join([f'{label}={prob:.3f}' for label, prob in top_probs])}")

# --- Cross-domain Evaluation (Mapping Labels) ---
print("\n" + "=" * 60)
print("CROSS-DOMAIN EVALUATION (Label Mapping)")
print("=" * 60)
print("\nMapping Jigsaw predictions to Davidson labels:")
print("  toxic/severe_toxic/identity_hate -> hate_speech")
print("  obscene/insult -> offensive_language")
print("  (low scores) -> neither")

# Create mapped predictions
hate_speech_pred = (probs_positive_class[:, [0, 1, 5]].max(axis=1) >= 0.5).astype(int)  # toxic, severe_toxic, identity_hate
offensive_pred = (probs_positive_class[:, [2, 4]].max(axis=1) >= 0.5).astype(int)  # obscene, insult
neither_pred = ((1 - probs_positive_class.max(axis=1)) >= 0.5).astype(int)

y_pred_mapped = np.column_stack([hate_speech_pred, offensive_pred, neither_pred])

print("\nMapped prediction distribution:")
for i, label in enumerate(existing_davidson_labels):
    positive_preds = y_pred_mapped[:, i].sum()
    print(f"  {label}: {positive_preds} positive predictions ({100*positive_preds/len(y_pred_mapped):.1f}%)")

# --- Evaluation Metrics ---
print("\n" + "=" * 60)
print("CROSS-DOMAIN EVALUATION METRICS (Mapped Labels)")
print("=" * 60)

for i, label in enumerate(existing_davidson_labels):
    print(f"\n--- Metrics for label: {label} ---")
    print(f"  Accuracy:  {accuracy_score(y_test.iloc[:, i], y_pred_mapped[:, i]):.4f}")
    print(f"  Precision: {precision_score(y_test.iloc[:, i], y_pred_mapped[:, i], zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test.iloc[:, i], y_pred_mapped[:, i], zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test.iloc[:, i], y_pred_mapped[:, i], zero_division=0):.4f}")

print("\n--- Overall Metrics ---")
print(f"  Hamming Loss:         {hamming_loss(y_test, y_pred_mapped):.4f}")
try:
    print(f"  Jaccard Score (avg):  {jaccard_score(y_test, y_pred_mapped, average='samples'):.4f}")
except:
    print(f"  Jaccard Score (avg):  N/A")

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
