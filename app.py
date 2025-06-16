import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Page Title
st.title("🧠 Interactive ML Model Playground")
st.write("Upload your CSV datasheet, select your target column and models, and see results instantly")

#Upload CSV
uploaded_file = st.file_uploader("Upload your datasheet (.csv)", type=["csv"])
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write("### Preview of Dataset", df.head())
#Select target column
target_col = st.selectbox("Select the target column", df.columns)

#Select models
model_options = ["LogisticRegression", "DecisionTree", "RandomForest", "KNN"]
selected_models = st.multiselect("Choose ML models to train", model_options)

if selected_models and target_col:
  #Preprocessing
  X = df.drop(columns=[target_col])
  y = df[target_col]

  #Handle non-numeric columns (basic)
  X = pd.get_dummies(X)

  #Split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

  # Model training
  st.write("### Results ###")
  for model_name in selected_models:
    if model_name == "LogisticRegression":
      model = LogisticRegression(max_iter=1000)
    elif model_name == "DecisionTree":
      model = DecisionTreeClassifier()
    elif model_name == "RandomForest":
      model = RandomForestClassifier()
    elif model_name == "KNN":
      model = KNeighborsClassifier()

      model.fit(X_train, y_train)
      predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    st.write(f"### {model_name}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions))




