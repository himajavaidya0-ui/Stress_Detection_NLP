# Stress Detection from Text using NLP

A machine learning project that classifies text as **Stress** or **No Stress** using Natural Language Processing (NLP) and machine learning techniques.

## Project Overview

This project processes text data using NLP preprocessing techniques, converts the cleaned text into numerical features using **TF-IDF**, and evaluates multiple machine learning classification algorithms.

The trained model can also accept new user-provided text and predict whether it belongs to the **Stress** or **No Stress** category.

## Workflow

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Stopword Removal & Stemming
   ↓
Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning Models
   ├── Bernoulli Naive Bayes
   ├── Logistic Regression
   └── Linear SVM
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Stress / No Stress Prediction
````

## Dataset

The dataset contains **2,838 text samples** and 116 features.

Target distribution:

* **Stress (1):** 1,488 samples
* **No Stress (0):** 1,350 samples

For classification, the primary columns used are:

* `text` — input text
* `label` — target class

The dataset contains no missing values.

## Text Preprocessing

The text preprocessing pipeline includes:

* Converting text to lowercase
* Removing URLs
* Removing HTML tags
* Removing punctuation
* Removing numbers
* Removing stopwords
* Applying **Snowball Stemming**
* Removing extra whitespace

## Feature Extraction

### TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** was used to convert cleaned text into numerical feature vectors.

A maximum of **5,000 features** was used for model training.

The TF-IDF vectorizer was fitted only on the training data and then used to transform the test data. This prevents information from the test set from influencing the feature extraction process.

## Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

The split used `stratify` to maintain approximately the same proportion of Stress and No Stress samples in both sets.

A fixed `random_state=42` was used to ensure reproducibility.

## Machine Learning Models

Three classification algorithms were trained and evaluated:

1. **Bernoulli Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (SVM)**

These models were selected to compare different approaches to binary text classification using TF-IDF features.

## Model Performance

The models were evaluated using **Accuracy, Precision, Recall, and F1 Score**.

| Model                 |   Accuracy |  Precision |     Recall |   F1 Score |
| --------------------- | ---------: | ---------: | ---------: | ---------: |
| Bernoulli Naive Bayes |     74.65% |     72.78% | **82.55%** | **77.36%** |
| Logistic Regression   | **76.06%** | **77.18%** |     77.18% |     77.18% |
| Linear SVM            |     74.12% |     75.08% |     75.84% |     75.46% |

> Precision, Recall, and F1 Score shown above refer to the **Stress class (label 1)**.

### Final Model

**Logistic Regression** was selected as the final model because it achieved the highest overall accuracy among the tested models and provided balanced precision and recall for the Stress class.

Bernoulli Naive Bayes achieved a higher recall for the Stress class, meaning it identified a larger proportion of actual stress-related samples. However, Logistic Regression provided better overall accuracy and higher precision while maintaining a comparable F1 score.

## Confusion Matrix

The Logistic Regression model produced the following confusion matrix on the test set:

```text
                 Predicted
              No Stress   Stress

Actual
No Stress        202         68
Stress            68        230
```

The confusion matrix represents:

* **True Negative (TN):** 202
* **False Positive (FP):** 68
* **False Negative (FN):** 68
* **True Positive (TP):** 230

The model correctly classified **432 out of 568 test samples**, resulting in an accuracy of approximately **76.06%**.

## Prediction on New Text

The project includes a prediction function that accepts new user-provided text and applies the same preprocessing and TF-IDF transformation used during training.

Example:

```text
Input:
I have been feeling overwhelmed with everything lately.

Output:
Stress
```

## Project Structure

```text
Stress_Detection_NLP/
│
├── data/
│   └── stress.csv
│
├── models/
│   ├── stress_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── .gitignore
├── requirements.txt
├── Stress_Detection.ipynb
└── README.md
```

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **NLTK**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **WordCloud**
* **Joblib**
* **Jupyter Notebook**

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Stress_Detection_NLP.git
cd Stress_Detection_NLP
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Git Bash:**

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

Open `Stress_Detection.ipynb` in VS Code or Jupyter Notebook and run the cells.

## Saved Models

The trained Logistic Regression model and fitted TF-IDF vectorizer are saved using **Joblib**:

```text
models/stress_model.pkl
models/tfidf_vectorizer.pkl
```

Saving these components allows the trained model and feature extraction pipeline to be reused without retraining.

## Future Improvements

* Hyperparameter tuning
* Larger and more diverse datasets
* Advanced NLP feature engineering
* Transformer-based NLP models
* Streamlit or Flask deployment
* Confidence-based predictions
* Model explainability
* Improved text classification performance

## Disclaimer

This project is intended for **educational and demonstration purposes only**.

It is a machine learning text-classification project and should **not** be used as a medical or psychological diagnostic tool.

