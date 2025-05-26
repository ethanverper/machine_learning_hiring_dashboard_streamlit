
# 🧠 Hiring Strategy Dashboard – Logistic vs Decision Tree

This Streamlit dashboard compares the performance of two models (Logistic Regression and Decision Tree) in predicting hiring outcomes based on candidate features and recruitment strategies.

## 🔍 Features

- Interactive model training (Logistic and Decision Tree)
- Side-by-side evaluation with:
  - Accuracy
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall
  - Residuals
  - Cross-validation
- Custom prediction interface
- Executive summary generation
- Password-protected access (via Streamlit secrets)

## 📁 Project Structure

```
├── streamlit_experiment_ml_dashboard.py   # Main Streamlit app
├── recruitment_data.csv                  # Dataset for model training
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml (excluded)
├── requirements.txt
├── README.md
```

## 🚀 Getting Started

1. Clone the repo:
```
git clone https://github.com/your_user/your_repo.git
cd your_repo
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the dashboard:
```
streamlit run streamlit_experiment_ml_dashboard.py
```

---

Developed by Ethan Verduzco – [LinkedIn]([https://www.linkedin.com](https://www.linkedin.com/in/ethanverper/))
