# 📊 Lending Club Loan Default Prediction

This project focuses on predicting the likelihood of loan default using the **Lending Club dataset** (Kaggle).  
By applying data preprocessing, feature engineering, and machine learning models, the system helps financial institutions make better lending decisions.  
Additionally, a **Gradio-based application** is developed to provide a user-friendly interface for loan default prediction.

---

## 🚀 Features
- Data preprocessing:
  - Handling missing values, outliers, and data cleaning.
  - Encoding categorical variables using **OrdinalEncoder** and **OneHotEncoder**.
  - Feature scaling with **MinMaxScaler**.
- Machine learning:
  - Built and trained a **Random Forest Classifier** for loan default prediction.
  - Stored trained models and preprocessing objects using **joblib**.
- Deployment:
  - Gradio web application for interactive loan default prediction.
  - User inputs borrower details and receives prediction with probability score.

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ducthien-05/Lending-Club.git
   cd Lending-Club
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   source .venv/bin/activate # On Linux/Mac
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## ▶️ Usage
1. Train the model (if needed):
   ```bash
   python src/train_model.py
2. Run the Gradio application:
   ```bash    
   python src/predict_app.py
3. Access the app:
   Local: http://***.*.*.*:****
   Public link (optional): Generated if share=True works

---
## 🛠️ Tech Stack
  - Python
  - Pandas, Scikit-learn, Numpy, Searbon, Matplotlib
  - Gradio (for deployment)
  - Joblib (for model persistence)
