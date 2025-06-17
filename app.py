import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Page config
st.set_page_config(page_title = "ML Playground", layout = "centered")
st.title("🧠 Interactive ML Model Playground")

uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 Preview of your data", df.head())
    target_col = st.selectbox("🎯 Select target column", df.columns)
    feature_cols = [col for col in df.columns if col != target_col]
    #Random Forest Trees Slider (always visible for now)
    rf_n_estimators = st.slider("🌲 Number of trees for Random Forest", 10, 500, 100)
    #Model choices
    st.markdown("## 🤖 Choose up to 5 models")
    selected_models = st.multiselect(
        "Select ML models to train or combine:",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "KNN", "SVM"]
    )
    if st.button("🚀 Run Model(s)"):
        if len(selected_models) == 0:
            st.warning("Please select at least one model.")
        elif len(selected_models) > 5:
            st.error("❌ You can select up to 5 models only.")
        else:
            X = df[feature_cols]
            y = df[target_col]
            X = pd.get_dummies(X)
            X = X.dropna()
            y = y.loc[X.index]
            if y.dtype == 'object':
                y = pd.factorize(y)[0]
            if y.isnull().sum() > 0:
                sr.error("❌ Target column contains missing values.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model_map = {
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(n_estimators=rf_n_estimators),
                    "Naive Bayes": GaussianNB(),
                    "KNN": KNeighborsClassifier(),
                    "SVM": SVC(probability=True),
                }
                if len(selected_models) == 1:
                    model = model_map[selected_models[0]]
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    st.success(f"✅ {selected_models[0]} Accuracy: {round(acc*100, 2)}%")
                else:
                    try:
                        ensemble = VotingClassifier(
                            estimators = [(name, model_map[name]) for name in selected_models],
                            voting = 'hard'
                        )
                        ensemble.fit(X_train, y_train)
                        preds = ensemble.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        names = ", ".join(selected_models)
                        st.success(f"✅ Ensemble of [{names}] Accuracy: {round(acc*100, 2)}%")
                    except Exception as e:
                        st.error(f"⚠️ Cannot combine selected models. Error: {str(e)}")
                        st.info("💡 Tip: Avoid mixing too many different types of models. Try 2-3 simpler ones like Logistic Regression, Naive Bayes, or KNN.")
else:
    st.info("⬆️ Upload a CSV file to get started.")
