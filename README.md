# 🧠 Interactive ML Model Playground

This is a Streamlit-based web application that allows users to upload their own datasets and train a variety of classical machine learning models — all directly in the browser. No coding required!

---

## 🚀 Features

- 📂 Upload any `.csv` dataset
- 🎯 Select your target column
- ✅ Automatic preprocessing with one-hot encoding
- 🤖 Choose from classic ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest (customizable tree count)
  - Naive Bayes
  - K-Nearest Neighbors
  - Support Vector Machine (SVM)
- 🔀 Combine up to 5 models into an ensemble using `VotingClassifier`
- 📊 View accuracy results instantly after training

---

## 📦 Tech Stack

- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/) for UI

---

## 🧪 How to Use

1. Clone the repo or visit the live app:  
   👉 [Live Demo](https://your-app-link.streamlit.app) *(replace with actual link)*

2. Upload a `.csv` file with structured tabular data.

3. Select the **target column** (the label you want to predict).

4. Choose one or more ML models from the list.

5. If using **Random Forest**, adjust the number of trees using the slider.

6. Hit **Run** and watch the magic happen.

---

## 💡 Notes

- The app handles basic preprocessing (encoding categorical variables, dropping rows with missing data).
- If the target column is non-numeric, it will be automatically encoded.
- Some models (like Logistic Regression or SVM) may take longer depending on the dataset.
- Naive Bayes may underperform on datasets with many encoded categorical features (like Titanic).

---

## 🧰 Example Datasets

Try it out with:
- [`titanic.csv`](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- Any CSV file with numeric and categorical features

---

## 📄 License

This project is for educational purposes. Feel free to fork, customize, or build on top of it.

