# SMS Spam Classifier

A machine learning project that classifies SMS text messages as **Spam** or **Ham (Not Spam)** using classic NLP feature extraction and multiple classification algorithms - Naive Bayes, Support Vector Machines (Linear & Sigmoid kernels), K-Nearest Neighbors, and Logistic Regression.

## 📌 Overview

This project uses the well-known **SMS Spam Collection Dataset** to build and compare text-classification models. Raw SMS messages are converted into numerical features using `CountVectorizer` (Bag-of-Words), and several supervised learning algorithms are trained and evaluated on their ability to correctly flag spam messages.

## 📂 Dataset

- **File:** `spam.csv`
- **Columns:**
  - `v1` — Label (`ham` or `spam`)
  - `v2` — Raw SMS message text
- **Size:** 5,572 messages
- The dataset is the standard "SMS Spam Collection" dataset (UCI Machine Learning Repository / commonly distributed on Kaggle).

> **Note:** `spam.csv` is not included in this repository due to size/licensing conventions. Download it from [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project root before running the notebook.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `numpy`, `pandas` - data handling
  - `matplotlib`, `seaborn` - visualization
  - `scikit-learn` - feature extraction & machine learning models
  - `collections.Counter` - word frequency analysis

## 🔍 Project Workflow

1. **Data Loading** - Read `spam.csv` into a pandas DataFrame.
2. **Data Cleaning** - Check and handle missing/null values.
3. **Exploratory Data Analysis (EDA)**
   - Most frequent words in spam vs. ham messages (bar charts)
   - Class distribution plot (`ham` vs `spam` count)
4. **Feature Engineering**
   - Convert text to numeric vectors using `CountVectorizer` (with English stop-word removal)
   - Encode labels: `ham → 0`, `spam → 1`
5. **Train/Test Split** - 80% training, 20% testing (`random_state=42`)
6. **Model Training & Evaluation** - Train and evaluate four classifiers:
   - Multinomial Naive Bayes
   - Support Vector Machine (Linear kernel)
   - Support Vector Machine (Sigmoid kernel)
   - K-Nearest Neighbors
   - Logistic Regression
7. **Performance Comparison** - Accuracy score, classification report (precision/recall/F1), and confusion matrix for each model.

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | **98.03%** |
| SVM (Linear kernel) | 98.03% |
| SVM (Sigmoid kernel) | **98.12%** |
| K-Nearest Neighbors | 91.75% |
| Logistic Regression | 97.85% |

The **SVM with Sigmoid kernel** achieved the highest accuracy, closely followed by Naive Bayes and the Linear SVM. KNN underperformed relative to the other models, particularly on spam recall.

## 🚀 Getting Started

### Prerequisites
Make sure Python 3.8+ is installed, then install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/sms-spam-classifier.git
   cd sms-spam-classifier
   ```
2. Add the dataset `spam.csv` to the project root (see [Dataset](#-dataset) section above).
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `Spam_message_NB.ipynb` and run all cells sequentially.

## 📁 Project Structure

```
sms-spam-classifier/
│
├── Spam_message_NB.ipynb   # Main notebook: EDA, feature engineering, model training & evaluation
├── spam.csv                # Dataset (not included — see Dataset section)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## 📈 Future Improvements

- Use TF-IDF vectorization instead of / alongside raw Count Vectorization
- Try ensemble methods (Random Forest, Gradient Boosting, XGBoost)
- Hyperparameter tuning via GridSearchCV
- Build a simple web app (Flask/Streamlit) for real-time spam prediction
- Handle class imbalance with techniques like SMOTE

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](../../issues) or submit a pull request.

## 🙋 Author

Created as a Natural Language Processing / Machine Learning mini-project for SMS spam detection.
