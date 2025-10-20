---

# 🧠 AI & Machine Learning Projects  
### Scikit-learn | PyTorch | spaCy  

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?logo=scikitlearn)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch)
![spaCy](https://img.shields.io/badge/spaCy-green?logo=spacy)
![Jupyter](https://img.shields.io/badge/Notebook-Interactive-orange?logo=jupyter)

This repository contains **three end-to-end projects** showcasing Machine Learning, Deep Learning, and Natural Language Processing using some of the most powerful Python libraries — **Scikit-learn, PyTorch, and spaCy**.  
Each project is presented both as an individual Python script and as part of a unified **Jupyter Notebook** (`week3ML_DL_NLP.ipynb`) with corresponding outputs.

---

## 📂 Project Structure

📦 AI-ML-Projects │ ├── 📘 week3ML_DL_NLP.ipynb        # Combined notebook with all tasks ├── 🧩 machineLearning-sklearn.py             # Classical ML: Decision Tree Classifier ├── 🔥 deeplearning-pytorch.py                       # Deep Learning: CNN on MNIST ├── 💬 nlp_spaCy.py                         # NLP: Entity Recognition & Sentiment ├── 📸 images/                         # Output images and graphs └── README.md                               # Documentation (this file)

---

## 1️⃣ Classical Machine Learning – Decision Tree (Scikit-learn)  

**Dataset:** *Iris Flower Dataset (UCI Repository)*  
**Objective:** Predict the species of an iris flower based on petal and sepal dimensions.

### 🧾 Workflow
1. Load dataset and explore using Pandas.  
2. Encode labels and split into training/testing sets.  
3. Train a **Decision Tree Classifier**.  
4. Evaluate with **accuracy**, **precision**, **recall**, and **confusion matrix**.  
5. Visualize decision boundaries and tree structure.

### 📈 Results
| Metric | Score |
|--------|--------|
| Accuracy | **≈ 0.89** |
| Precision | **≈ 0.90** |
| Recall | **≈ 0.90** |

### 🖼️ Screenshots
![Confusion Matrix](images/scikit-learn%20model/confusionMatrix.png)
![Decision Tree Visualization](images/scikit-learn%20model/decisiontree.png)

---

## 2️⃣ Deep Learning – CNN for MNIST Digit Recognition (PyTorch)  

**Dataset:** *MNIST Handwritten Digits (28×28 grayscale images)*  
**Objective:** Build a **Convolutional Neural Network (CNN)** that classifies digits 0–9 with >95% accuracy.

### ⚙️ Architecture

Input (1x28x28) → Conv2D(1→16, kernel=3) → ReLU → MaxPool(2x2) → Conv2D(16→32, kernel=3) → ReLU → MaxPool(2x2) → Flatten → Linear(3255→128) → ReLU → Linear(128→10) → Softmax

### 🔧 Training Parameters
- Optimizer: **Adam**
- Learning Rate: **0.0005**
- Epochs: **5**
- Loss Function: **CrossEntropyLoss**

### 🧠 Training Summary
| Epoch | Loss | Accuracy |
|-------|------|-----------|
| 1 | 0.1977 | 94.17% |
| 2 | 0.0572 | 98.25% |
| 3 | 0.0409 | 98.75% |
| 4 | 0.0300 | 99.08% |
| 5 | 0.0234 | 99.23% |

✅ **Final Test Accuracy:** 99.03%  
📦 **Model Saved:** `cnn_mnist_model.pth`

### 📸 Screenshots
![Training Accuracy vs Loss](images/pyTorch%20Model/loss-AccuracyVSepochs.png)
![Sample Predictions](images/pyTorch%20Model/sample1.png)

---

## 3️⃣ Natural Language Processing – Entity Recognition & Sentiment (spaCy)  

**Dataset:** *Amazon Product Reviews (Kaggle – train.ft.txt)*  
**Objective:** Extract **brand and product entities** and perform **rule-based sentiment analysis**.

### 🧩 Implementation Steps
1. Load the first 100,000 reviews from the dataset (~1.5 GB total).  
2. Process text with `spaCy`’s `en_core_web_sm` model.  
3. Extract entities labeled as `ORG` (brands) and `PRODUCT`.  
4. Apply a **custom rule-based sentiment analyzer** using predefined positive and negative lexicons.

### 💬 Example Output

Review 1: Excellent sound quality from these Sony speakers. Sentiment: positive Entities: [('Sony', 'ORG')]

Review 2: The charger broke after two days. Poor quality. Sentiment: negative Entities: [('charger', 'PRODUCT')]

### 📸 Screenshots
![Entity Recognition](images/spaCy%20Model/examplesofSentiment.png)
![Sentiment Output](images/spaCy%20Model/examplesofSentiment.png)

---

## 📓 Combined Notebook  

All three tasks — Scikit-learn, PyTorch, and spaCy — are organized and executed in  
**`week3ML_DL_NLP.ipynb`**, containing:
- Explanatory markdown cells  
- Code cells with outputs  
- Visualizations and metrics  

This makes it easy to follow each experiment interactively.

---

## 🧰 Tools & Technologies  

| Category | Tools |
|-----------|--------|
| Language | Python 3.x |
| ML Framework | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | spaCy |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| IDE | Jupyter Notebook |

---

## 🧠 Key Learnings  

- Hands-on experience with data preprocessing, feature encoding, and model evaluation.  
- Practical implementation of **CNNs** for computer vision tasks.  
- Use of pre-trained NLP pipelines for **Named Entity Recognition**.  
- Designing a **custom sentiment analysis algorithm** without external libraries.  
- Saving, loading, and interpreting trained models.

---

## 🚀 Future Enhancements  

- Integrate **GridSearchCV** for hyperparameter optimization (Scikit-learn).  
- Add **Dropout** and **Batch Normalization** layers for better CNN generalization.  
- Replace rule-based sentiment with **VADER** or **BERT-based models**.  
- Deploy as a **Streamlit or Flask web app** for interactive demos.  

---

## 👨‍💻 Author  

**Erick Wambugu**  
Power Learn Project Software Development Scholar  

📧 **Email:** erick.wambugu23@gmail.com 
🌐 **Portfolio:** erickwambugu.netlify.app 
💼 **LinkedIn:** https://www.linkedin.com/in/erick-wambugu-425a15161/

---

> *“Artificial Intelligence is not magic — it’s data, logic, and curiosity applied consistently.”*  
> — *Erick, Meteorologist, AI, ML & Data Enthusiast*  

---


---


