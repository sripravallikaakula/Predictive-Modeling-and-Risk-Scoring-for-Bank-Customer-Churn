import requests
import pandas as pd
import streamlit as st
import plotly.express as px


API_URL="https://predictive-modeling-and-risk-scoring-for-bank-cu-production.up.railway.app"

def call_api(endpoint, method="GET", payload=None):
    try:
        url = f"{API_URL.rstrip('/')}/{endpoint.lstrip('/')}"

        if method.upper() == "POST":
            response = requests.post(
                url,
                json=payload,
                timeout=30,
            )
        else:
            response = requests.get(
                url,
                timeout=30,
            )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to the Railway backend. "
            "Please check that the Railway service is running."
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "⏱️ The Railway backend took too long to respond."
        )
        return None

    except requests.exceptions.HTTPError as exc:
        try:
            detail = (
                response.json().get("detail", str(exc))
                if response is not None
                else str(exc)
            )
        except Exception:
            detail = str(exc)

        st.error(f"❌ API error: {detail}")
        return None

    except Exception as exc:
        st.error(f"❌ Unexpected error: {exc}")
        return None


def customer_form(prefix="main"):
    """
    Collects both:
    1. Customer personal/identification details
    2. ML prediction features

    CustomerId, Surname and Year are NOT sent as model features.
    They are only used for identifying/displaying the customer.
    """

    st.markdown("### 👤 Customer Personal Details")

    personal_col1, personal_col2, personal_col3 = st.columns(3)

    with personal_col1:
        customer_id = st.number_input(
            "Customer ID",
            min_value=1,
            value=15634602,
            step=1,
            key=f"{prefix}_customer_id",
        )

    with personal_col2:
        surname = st.text_input(
            "Surname",
            value="Hargrave",
            key=f"{prefix}_surname",
        )

    with personal_col3:
        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1,
            key=f"{prefix}_year",
        )

    st.markdown("### 🏦 Customer Banking & Demographic Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            key=f"{prefix}_credit",
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Spain", "Germany"],
            key=f"{prefix}_geo",
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key=f"{prefix}_gender",
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40,
            key=f"{prefix}_age",
        )

    with col2:
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=20,
            value=5,
            key=f"{prefix}_tenure",
        )

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=75000.0,
            step=1000.0,
            key=f"{prefix}_balance",
        )

        products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=2,
            key=f"{prefix}_products",
        )

    with col3:
        card = st.selectbox(
            "Has Credit Card?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key=f"{prefix}_card",
        )

        active = st.selectbox(
            "Active Member?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key=f"{prefix}_active",
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            key=f"{prefix}_salary",
        )

    return {
        "CustomerId": int(customer_id),
        "Surname": surname.strip(),
        "Year": int(year),
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(products),
        "HasCrCard": int(card),
        "IsActiveMember": int(active),
        "EstimatedSalary": float(salary),
    }
    

def main():
    global API_URL

    st.set_page_config(
        page_title="European Bank Churn Intelligence",
        page_icon="🏦",
        layout="wide",
    )

    API_URL = st.sidebar.text_input(
    "FastAPI URL",
    value="https://predictive-modeling-and-risk-scoring-for-bank-cu-production.up.railway.app",
).rstrip("/")
    

    st.title("🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn")
    st.caption(
        "Customer-level churn probability, risk classification, explainability, "
        "and what-if scenario analysis."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Risk Calculator",
            "Model Performance",
            "Feature Importance",
            "What-If Simulator",
        ]
    )

    with tab1:
        st.subheader("👤 Customer Churn Risk Calculator")

        payload = customer_form("main")

        st.divider()

        if st.button(
            "🔮 Predict Churn Risk",
            type="primary",
            use_container_width=True,
        ):
            result = call_api(
                "/predict",
                method="POST",
                payload=payload,
            )

            if result:
                st.success("Customer details received successfully.")
                st.markdown("### 👤 Customer Information")

                customer_col1, customer_col2, customer_col3 = st.columns(3)
                customer_col1.metric(
                    "Customer ID",
                    result.get("customer", {}).get("CustomerId", payload["CustomerId"]),
                )
                customer_col2.metric(
                    "Surname",
                    result.get("customer", {}).get("Surname", payload["Surname"]),
                )
                customer_col3.metric(
                    "Year",
                    result.get("customer", {}).get("Year", payload["Year"]),
                )

                st.markdown("### 📊 Churn Prediction")
                c1, c2, c3 = st.columns(3)
                c1.metric("Churn Probability", f"{result['churn_probability_percent']:.2f}%")
                c2.metric("Risk Level", result["risk_level"])
                c3.metric("Prediction", result["prediction"])

                probability = result["churn_probability"]
                chart_df = pd.DataFrame(
                    {
                        "Outcome": ["Churn", "Retain"],
                        "Probability": [probability, 1 - probability],
                    }
                )

                fig = px.bar(
                    chart_df,
                    x="Outcome",
                    y="Probability",
                    range_y=[0, 1],
                    title="Churn Probability Distribution",
                    labels={"Probability": "Probability", "Outcome": "Customer Outcome"},
                    text_auto=".2f",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Customer Details")
                details_df = pd.DataFrame(
                    {
                        "Field": [
                            "Customer ID",
                            "Surname",
                            "Year",
                            "Credit Score",
                            "Geography",
                            "Gender",
                            "Age",
                            "Tenure",
                            "Balance",
                            "Number of Products",
                            "Has Credit Card",
                            "Active Member",
                            "Estimated Salary",
                        ],
                        "Value": [
                            payload["CustomerId"],
                            payload["Surname"],
                            payload["Year"],
                            payload["CreditScore"],
                            payload["Geography"],
                            payload["Gender"],
                            payload["Age"],
                            payload["Tenure"],
                            f"{payload['Balance']:,.2f}",
                            payload["NumOfProducts"],
                            "Yes" if payload["HasCrCard"] == 1 else "No",
                            "Yes" if payload["IsActiveMember"] == 1 else "No",
                            f"{payload['EstimatedSalary']:,.2f}",
                        ],
                    }
                )
                st.dataframe(details_df, use_container_width=True, hide_index=True)

                st.subheader("💡 Recommended Retention Actions")
                for item in result["recommendations"]:
                    st.write("•", item)

    with tab2:
        st.subheader("📈 Model Evaluation")

        if st.button("Load Model Metrics", use_container_width=True):
            info = call_api("/model-info")
            if info:
                st.success(f"Selected model: {info['best_model']}")
                metrics = pd.DataFrame(info["results"]).T
                st.dataframe(metrics, use_container_width=True)

                if "roc_auc" in metrics.columns:
                    fig = px.bar(
                        metrics.reset_index(),
                        x="index",
                        y="roc_auc",
                        title="ROC-AUC by Model",
                        labels={"index": "Model", "roc_auc": "ROC-AUC"},
                        text_auto=".3f",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📊 Feature Importance Dashboard")

        if st.button("Load Feature Importance", use_container_width=True):
            result = call_api("/feature-importance?limit=15")
            if result and result.get("available"):
                importance_df = pd.DataFrame(result["items"])
                fig = px.bar(
                    importance_df.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Churn Drivers",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(importance_df, use_container_width=True)
            elif result:
                st.info(result.get("message", "Feature importance unavailable."))

    with tab4:
        st.subheader("🔄 What-If Scenario Simulator")
        st.write(
            "Change customer engagement or product usage and observe how the predicted churn probability changes."
        )

        scenario = customer_form("scenario")

        if st.button("🚀 Run What-If Scenario", use_container_width=True):
            result = call_api("/what-if", method="POST", payload=scenario)

            if result:
                st.markdown("### Customer")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Customer ID",
                    result.get("customer", {}).get("CustomerId", scenario["CustomerId"]),
                )
                c2.metric(
                    "Surname",
                    result.get("customer", {}).get("Surname", scenario["Surname"]),
                )
                c3.metric(
                    "Year",
                    result.get("customer", {}).get("Year", scenario["Year"]),
                )

                st.markdown("### Scenario Result")
                st.metric("Scenario Churn Probability", f"{result['churn_probability_percent']:.2f}%")
                st.write("Risk Level:", result["risk_level"])
                st.write("Prediction:", result["prediction"])


if __name__ == "__main__":
    main()
