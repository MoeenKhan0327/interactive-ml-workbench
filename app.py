import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Title
st.set_page_config(page_title="Interactive ML Playground", layout="centered")
st.title("🧠 Interactive ML Model Playground")
st.markdown("Upload your CSV datasheet, select your target column and models, and see results instantly")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview of your dataset:", df.head())

    # Select target column
    target_col = st.selectbox("🎯 Select the target column", df.columns)

    # Get feature columns (auto-detect)
    feature_cols = [col for col in df.columns if col != target_col]

    # Model selection
    st.markdown("## 🤖 Choose your models")
    models_selected = st.multiselect(
        "Select one or more ML models to train:",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "KNN", "SVM"]
    )

    if st.button("🚀 Train Selected Model(s)") and models_selected:
        # Prepare data
        X = df[feature_cols]
        y = df[target_col]

        # Handle categorical features (basic handling)
        X = pd.get_dummies(X)
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_scores = {}

        for model_name in models_selected:
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "KNN":
                model = KNeighborsClassifier()
            elif model_name == "SVM":
                model = SVC()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            model_scores[model_name] = acc

        st.markdown("## 📊 Model Accuracy")
        for name, score in model_scores.items():
            st.write(f"**{name}**: {round(score * 100, 2)}%")

else:
    st.warning("⬆️ Please upload a CSV file to begin.")
