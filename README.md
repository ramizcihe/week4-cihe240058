"# week4-cihe240058" 

# 🛡️ Cyberbullying Detection Pipeline

This project involves building a cyberbullying detection system using natural language processing (NLP) and machine learning techniques. Below is a step-by-step breakdown of the tasks performed:

---

## 📂 1. Dataset Preparation
- Imported the cyberbullying dataset from Kaggle or other public sources.
- Explored dataset structure and label distribution.
- Identified class imbalance.
- Cleaned the dataset by removing:
  - Missing/null values
  - Duplicate entries
  - Irrelevant content (e.g., advertisements, unrelated text)

---

## 🧹 2. Text Preprocessing
- Converted all text to lowercase.
- Removed:
  - Stopwords
  - Punctuation
  - Numbers and special characters
- Tokenized the text into individual words.
- Applied stemming or lemmatization.
- Optionally removed very rare or overly frequent words to reduce noise.

---

## 🔍 3. Feature Extraction & Selection
- Represented text using:
  - **TF-IDF** vectors (word-level, n-grams, character-level)
  - **Word Embeddings**: Word2Vec, GloVe, FastText
  - **Transformer-based Embeddings**: BERT, DistilBERT
- Selected relevant features using techniques like:
  - Chi-Square Test
  - Mutual Information

---

## 🤖 4. Model Selection
Explored and selected suitable classification models:
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Random Forest / Gradient Boosting
- Deep learning models:
  - LSTM
  - BiLSTM
  - Transformer-based models (e.g., BERT)

---

## 🧪 5. Dataset Splitting
- Split the dataset into:
  - Training set (70%)
  - Validation set (15%)
  - Test set (15%)
- Used **stratified splitting** to preserve class distribution across sets.

---

## 📊 6. Baseline Models
Implemented baseline models for comparison:
- Logistic Regression (with TF-IDF features)
- Naive Bayes (with TF-IDF features)
- SVM (with TF-IDF features)
- LSTM or BERT (with embedding-based input)

---

## 📈 7. Evaluation Metrics
Evaluated models using the following metrics:
- **Precision, Recall, F1-Score**
- **AUC-ROC**
- **Confusion Matrix** for detailed error analysis

---

## 🏋️ 8. Training & Evaluation
- Trained each model on the training dataset.
- Evaluated on the validation dataset.
- Performed hyperparameter tuning (e.g., learning rate, regularization strength).

---

## 📚 9. Model Comparison
- Created a summary table comparing:
  - Model performance (Precision, Recall, F1, AUC)
  - Training time and complexity
- Identified the best-performing model based on task goals and evaluation metrics.

---

## 📝 10. Documentation
Recorded all key components:
- Preprocessing techniques used
- Feature representation methods applied
- Model performance results
- Final model selection with justification
