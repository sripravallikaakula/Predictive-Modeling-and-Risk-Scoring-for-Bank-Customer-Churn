# Predictive Modeling and Risk Scoring for Bank Customer Churn

This project uses the supplied `European_Bank.csv` dataset and implements:

- Data preprocessing
- Feature engineering
- Stratified train/test split
- Logistic Regression
- Random Forest
- Gradient Boosting
- Optional XGBoost
- Accuracy, Precision, Recall, F1 and ROC-AUC evaluation
- Automatic best-model selection by ROC-AUC
- Churn probability scoring
- Low/Medium/High risk bands
- FastAPI backend
- Streamlit dashboard
- Feature importance
- What-if scenario simulation

## Project Structure

```text
bank_churn_project/
│
├── backend/
│   ├── app.py
│   ├── feature_engineering.py
│   └── train_model.py
│
├── frontend/
│   └── streamlit_app.py
│
├── data/
│   └── European_Bank.csv
│
├── models/
│   └── generated after training
│
├── requirements.txt
└── README.md
```

## 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

If XGBoost or SHAP causes an installation issue, you may temporarily remove those two
optional lines. The core project will still run with Logistic Regression, Random Forest
and Gradient Boosting.

## 3. Train the machine-learning models

From the project root:

```bash
python backend/train_model.py
```

This creates:

```text
models/churn_model.joblib
models/model_metadata.json
models/feature_importance.csv
```

The training script compares the available models and selects the model with the highest
test ROC-AUC.

## 4. Start the FastAPI backend

```bash
uvicorn backend.app:app --reload
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

Important API endpoints:

- `GET /health`
- `POST /predict`
- `POST /what-if`
- `GET /model-info`
- `GET /feature-importance`

## 5. Start the Streamlit frontend

Open a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The browser will normally open at:

```text
http://localhost:8501
```

## Prediction Input

The model accepts:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

`CustomerId`, `Surname`, and `Year` are not used as predictive features.

## Feature Engineering

The project adds:

- BalanceSalaryRatio
- ProductDensity
- EngagementProductInteraction
- AgeTenureInteraction

## Risk Score

The API returns the probability of churn from 0 to 1 and also as a percentage.

Default business risk bands:

- Low: less than 30%
- Medium: 30% to less than 60%
- High: 60% and above

The binary classification threshold defaults to 0.50.

## Note for Academic Project Use

This is a complete starting implementation for the project statement. For the final
research paper, report the actual evaluation metrics produced on your machine rather
than writing predetermined accuracy values.

Streamlit dashboard: http://127.0.0.1:8501
FastAPI backend: http://127.0.0.1:8000
API docs: http://127.0.0.1:8000/docs
