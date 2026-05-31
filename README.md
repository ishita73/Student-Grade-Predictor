# 🎓 Student Grade Predictor

A beginner-friendly Machine Learning project that predicts a student's final grade using academic performance indicators such as attendance, study hours, assignment scores, and previous GPA.

---

## 📖 Overview

Educational institutions often want to identify students who may need additional academic support before final examinations. This project uses a **Decision Tree Classifier** to analyze student data and predict the likely final grade.

The project demonstrates the complete Machine Learning workflow:

* Data Loading
* Data Cleaning
* Exploratory Data Analysis (EDA)
* Data Visualization
* Model Training
* Model Evaluation
* Grade Prediction

This project is designed specifically for students who are learning Machine Learning for the first time.

---

## 🚀 Features

✅ Load student data from CSV files

✅ Perform data cleaning and preprocessing

✅ Analyze student performance statistics

✅ Generate visualizations automatically

✅ Train a Decision Tree Classification model

✅ Evaluate model accuracy

✅ Predict grades for new students

✅ Display feature importance analysis

---

## 📂 Project Structure

```text
Student-Grade-Predictor/
│
├── data/
│   └── students.csv
│
├── outputs/
│   ├── data_analysis.png
│   └── feature_importance.png
│
├── docs/
│   └── Viva_QA.md
│
├── grade_predictor.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset Features

The model uses the following inputs:

| Feature         | Description                   |
| --------------- | ----------------------------- |
| Attendance      | Student attendance percentage |
| StudyHours      | Average daily study hours     |
| AssignmentScore | Assignment marks (0–100)      |
| PreviousGPA     | Previous semester GPA (0–10)  |

### Target Variable

| Output |
| ------ |
| A      |
| B      |
| C      |
| D      |
| F      |

---

## 🧠 Machine Learning Model

### Decision Tree Classifier

A Decision Tree was selected because:

* Easy to understand and visualize
* Suitable for classification tasks
* Requires minimal preprocessing
* Produces interpretable results
* Ideal for beginners learning ML concepts

The model learns patterns from student performance data and predicts the most likely final grade.

---

## 📈 Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Train-Test Split
   ↓
Decision Tree Training
   ↓
Model Evaluation
   ↓
Grade Prediction
```

---

## 🖼️ Results

### Data Analysis Dashboard

![Data Analysis](outputs/data_analysis.png)

### Feature Importance

![Feature Importance](outputs/feature_importance.png)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ishita73/Student-Grade-Predictor.git

cd Student-Grade-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python grade_predictor.py
```

The application will:

1. Load the dataset
2. Clean the data
3. Perform EDA
4. Generate visualizations
5. Train the model
6. Evaluate performance
7. Predict grades for new students

---

## 📌 Sample Prediction

```text
Attendance (%): 85
Study Hours: 5
Assignment Score: 78
Previous GPA: 7.5

Predicted Grade = B
```

---

## 📚 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Git
* GitHub

---

## 🎯 Learning Outcomes

Through this project, learners can understand:

* Data preprocessing techniques
* Exploratory Data Analysis (EDA)
* Classification algorithms
* Decision Trees
* Model evaluation metrics
* Feature importance
* Real-world ML project structure

---

## 🔮 Future Improvements

Potential enhancements include:

* Streamlit web application
* Student performance dashboard
* Multiple ML model comparison
* Model saving using Pickle
* CSV upload support
* Larger real-world datasets

---

## 👩‍💻 Author

**Ishita**

B.Tech Electronics & Communication Engineering (ECE)

Machine Learning & AI Enthusiast

---

## ⭐ If you found this project useful

Consider giving the repository a star and sharing it with other students learning Machine Learning.
