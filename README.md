# SMS Spam Detection System

This project implements and evaluates two distinct machine learning models for SMS spam detection: a traditional statistical model and a modern deep learning model. The goal is to compare their effectiveness in classifying messages as "spam" or "ham" (legitimate) using the well-known SMS Spam Collection Dataset.

The project demonstrates a complete NLP workflow, from data cleaning and preprocessing to model training, evaluation, and comparative analysis.

---

## 📋 Project Objectives

The primary objectives of this project are:

* **Establish a Baseline:** Implement a statistical model using **TF-IDF vectorization** with a **Naive Bayes classifier** to serve as a performance benchmark.
* **Implement an Embedding-Based Model:** Develop a **feedforward neural network** that leverages **pretrained Word2Vec embeddings** to capture deeper semantic meaning in the text.
* **Comparative Analysis:** Evaluate and compare the models on key performance metrics (Accuracy, Precision, Recall, F1-Score) to understand the trade-offs between the two approaches.

---

## 🛠️ Models & Methodology

### Data Preprocessing

Before training, all SMS messages undergo a rigorous cleaning process using NLTK:
1.  **Lowercasing & Normalization:** Text is converted to lowercase.
2.  **Tokenization:** Messages are broken down into individual words (tokens).
3.  **Stop-word Removal:** Common words with little semantic value (e.g., "the", "a", "is") are removed.
4.  **Lemmatization:** Words are reduced to their base or dictionary form (e.g., "running" becomes "run").

### 1. Baseline Model: TF-IDF + Naive Bayes

This model represents a classic, efficient approach to text classification.

* **Feature Extraction (TF-IDF):** The preprocessed text is converted into a numerical matrix using **Term Frequency-Inverse Document Frequency (TF-IDF)**. This method assigns weights to words based on their importance in a message relative to the entire dataset.
* **Classifier:** A **Multinomial Naive Bayes** classifier is trained on the TF-IDF features. This probabilistic model is fast, efficient, and performs well on high-dimensional text data.

### 2. Embedding-Based Model: Word2Vec + Neural Network

This model uses modern deep learning techniques to understand the context and semantics of words.

* **Feature Extraction (Word2Vec):** The pretrained **`word2vec-google-news-300`** model from Gensim is used to convert each word into a 300-dimensional vector. The vectors for all words in an SMS are then averaged to create a single feature vector for the entire message.
* **Classifier:** A **feedforward neural network**, built with TensorFlow and Keras, is trained on these embedding vectors. The architecture includes `Dense` layers with `relu` activation and `Dropout` layers to prevent overfitting.

---

## 📊 Performance Analysis

Both models were evaluated on a held-out test set (20% of the data). The Neural Network slightly outperformed the Naive Bayes model in overall accuracy and F1-score, primarily due to its significantly higher recall.

| Metric | Naive Bayes (TF-IDF) | Neural Network (Word2Vec) |
| :--- | :--- | :--- |
| **Accuracy** | 97.49% | **97.85%** |
| **Precision** | **99.19%** | 91.39% |
| **Recall** | 81.88% | **92.62%** |
| **F1-Score** | 89.71% | **92.00%** |

* **Naive Bayes** showed extremely high precision, making it excellent for applications where false positives are costly (e.g., corporate email filters).
* The **Neural Network** achieved much higher recall, making it better suited for tasks where catching as much spam as possible is the priority (e.g., moderating public content).

---

## 🚀 Technologies Used

* **Python 3**
* **TensorFlow & Keras**
* **Scikit-learn**
* **NLTK (Natural Language Toolkit)**
* **Gensim** (for Word2Vec)
* **Pandas & NumPy**
* **Matplotlib & Seaborn**

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

## ▶️ How to Run

The entire workflow is contained within the Jupyter Notebook.

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open and Run:**
    Open the `spam_detection.ipynb` file and run the cells sequentially. The script will load the data, train both models, and display the evaluation results and visualizations.

## 📚 References

[1] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. O'Reilly Media, 2009.

[2] D. Jurafsky and J. H. Martin, *Speech and Language Processing*. Pearson, 2021.

[3] T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering," *ACM Symposium on Document Engineering*, 2011.
