# 🎓 Student Grade Predictor
### A Beginner Machine Learning Project in Python

---

## 📌 Project Overview

The **Student Grade Predictor** is a beginner-friendly Machine Learning project that predicts a student's final grade (A / B / C / D / F) based on four key factors:

| Feature | Description |
|---|---|
| **Attendance (%)** | How regularly the student attends class |
| **Study Hours/Day** | Average daily study time |
| **Assignment Score** | Marks obtained in assignments (0–100) |
| **Previous GPA** | GPA from the previous semester (0–10) |

The model is trained using a **Decision Tree Classifier** from the scikit-learn library.

---

## 📁 Folder Structure

```
student_grade_predictor/
│
├── data/
│   └── students.csv          ← Dataset (60 student records)
│
├── outputs/
│   ├── data_analysis.png     ← Auto-generated graphs
│   └── feature_importance.png← Feature importance chart
│
├── grade_predictor.py        ← Main Python script (all-in-one)
└── README.md                 ← This file
```

---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ▶️ How to Run

```bash
cd student_grade_predictor
python grade_predictor.py
```

The program will:
1. Load and display the dataset
2. Clean the data (handle missing values, duplicates)
3. Show basic statistics (EDA)
4. Generate and save 4 graphs
5. Train the Decision Tree model
6. Show model accuracy and feature importance
7. Ask you to enter a student's data for prediction

---

## 🤖 Why Decision Tree?

| Reason | Explanation |
|---|---|
| **Visual** | You can literally draw the tree and understand decisions |
| **No scaling needed** | Unlike Linear Regression, features don't need normalization |
| **Classification** | Perfect for predicting categories (A, B, C, D, F) |
| **Explainable** | Every decision is traceable — great for learning |
| **Beginner-friendly** | Minimal math required to understand |

---

## 📊 Sample Output

```
========================================================
  STUDENT GRADE PREDICTOR - Loading Data
========================================================
✅ Dataset loaded successfully!
   Total students (rows): 60
   Total features (columns): 5

🎯 Model Accuracy: 91.67%

🏆 Predicted Final Grade: [ B ]
💬 Advice: Good performance! A little more effort can get you an A.
```

---

## 📚 Concepts Used

- **Pandas** — Loading CSV, data cleaning, groupby
- **NumPy** — Numerical operations
- **Matplotlib** — Bar charts, scatter plots, box plots
- **scikit-learn** — Decision Tree, train-test split, accuracy score
- **LabelEncoder** — Converting A/B/C to numbers for the model
- **train_test_split** — 80% train / 20% test split

---

## 🎯 Learning Outcomes

After completing this project, you will understand:
- How to load and explore real data
- What EDA (Exploratory Data Analysis) means
- How to train an ML model
- What accuracy, precision, and recall mean
- How to make predictions from a trained model

---

## 👩‍💻 Author

Built as a beginner ML project for B.Tech students.

---
