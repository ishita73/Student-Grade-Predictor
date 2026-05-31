# ============================================================
# PROJECT: Student Grade Predictor
# AUTHOR:  Your Name
# DATE:    2024
# MODEL:   Decision Tree Classifier
# ============================================================
# WHY DECISION TREE?
# - Easy to understand — works like a flowchart
# - No need to scale/normalize data
# - Great for classification problems (A, B, C, D, F)
# - You can actually see how it makes decisions
# - Perfect for beginners before moving to complex models
# ============================================================

# ---- STEP 1: Import Required Libraries ----
import pandas as pd          # For loading and handling data (like Excel in Python)
import numpy as np           # For numerical calculations
import matplotlib.pyplot as plt  # For drawing graphs
from sklearn.tree import DecisionTreeClassifier, export_text  # Our ML model
from sklearn.model_selection import train_test_split          # To split data
from sklearn.metrics import accuracy_score, classification_report  # To measure accuracy
from sklearn.preprocessing import LabelEncoder               # To convert grades to numbers
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings for clean output


# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================
def load_data(filepath):
    """
    Loads the CSV file into a pandas DataFrame.
    A DataFrame is like a table (similar to an Excel spreadsheet).
    """
    print("\n" + "="*60)
    print("  STUDENT GRADE PREDICTOR - Loading Data")
    print("="*60)

    data = pd.read_csv(filepath)

    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total students (rows): {data.shape[0]}")
    print(f"   Total features (columns): {data.shape[1]}")
    print(f"\n📋 First 5 rows of data:\n")
    print(data.head())
    return data


# ============================================================
# STEP 3: DATA CLEANING
# ============================================================
def clean_data(data):
    """
    Checks and cleans the data.
    Real-world data often has missing values or errors.
    We fix those here before training the model.
    """
    print("\n" + "="*60)
    print("  DATA CLEANING")
    print("="*60)

    # Check for missing values in each column
    missing = data.isnull().sum()
    print(f"\n🔍 Missing values in each column:")
    print(missing)

    # If any missing values exist, fill them with the column average
    if missing.sum() > 0:
        print("\n⚠️  Found missing values. Filling with column averages...")
        data = data.fillna(data.mean(numeric_only=True))
        print("✅ Missing values handled.")
    else:
        print("\n✅ No missing values found. Data is clean!")

    # Remove duplicate rows (if any student appears twice)
    before = len(data)
    data = data.drop_duplicates()
    after = len(data)
    if before != after:
        print(f"\n🗑️  Removed {before - after} duplicate rows.")
    else:
        print("✅ No duplicate rows found.")

    # Show data types of each column
    print(f"\n📊 Data types:\n{data.dtypes}")

    return data


# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def explore_data(data):
    """
    EDA means 'getting to know your data'.
    We look at statistics, patterns, and distributions.
    """
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    # Basic statistics: mean, min, max, etc.
    print("\n📈 Basic Statistics (Numerical Columns):\n")
    print(data.describe().round(2))

    # How many students got each grade?
    print("\n🎓 Grade Distribution (How many students per grade):\n")
    grade_counts = data['FinalGrade'].value_counts().sort_index()
    print(grade_counts)

    # Average study hours per grade
    print("\n📚 Average Study Hours by Grade:\n")
    avg_study = data.groupby('FinalGrade')['StudyHours'].mean().round(2)
    print(avg_study)


# ============================================================
# STEP 5: VISUALIZE THE DATA
# ============================================================
def visualize_data(data, output_folder="outputs"):
    """
    Creates 4 simple graphs to understand the data visually.
    Graphs help us see patterns that numbers alone can't show.
    """
    print("\n" + "="*60)
    print("  DATA VISUALIZATION")
    print("="*60)

    # Use a clean style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a figure with 4 subplots (2 rows x 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Student Grade Predictor - Data Analysis', fontsize=16, fontweight='bold', y=1.01)

    # --- Graph 1: Grade Distribution (Bar Chart) ---
    grade_counts = data['FinalGrade'].value_counts().sort_index()
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    axes[0, 0].bar(grade_counts.index, grade_counts.values, color=colors[:len(grade_counts)], edgecolor='black')
    axes[0, 0].set_title('Grade Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Final Grade')
    axes[0, 0].set_ylabel('Number of Students')
    for i, v in enumerate(grade_counts.values):
        axes[0, 0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

    # --- Graph 2: Attendance vs Final Grade (Box Plot) ---
    grades_order = sorted(data['FinalGrade'].unique())
    attendance_by_grade = [data[data['FinalGrade'] == g]['Attendance'].values for g in grades_order]
    bp = axes[0, 1].boxplot(attendance_by_grade, labels=grades_order, patch_artist=True)
    box_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], box_colors[:len(grades_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_title('Attendance % by Grade', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Final Grade')
    axes[0, 1].set_ylabel('Attendance (%)')

    # --- Graph 3: Study Hours vs Assignment Score (Scatter Plot) ---
    grade_colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12', 'D': '#e67e22', 'F': '#e74c3c'}
    for grade, group in data.groupby('FinalGrade'):
        axes[1, 0].scatter(
            group['StudyHours'], group['AssignmentScore'],
            label=f'Grade {grade}',
            color=grade_colors.get(grade, 'gray'),
            s=80, alpha=0.8, edgecolors='black', linewidths=0.5
        )
    axes[1, 0].set_title('Study Hours vs Assignment Score', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Study Hours per Day')
    axes[1, 0].set_ylabel('Assignment Score')
    axes[1, 0].legend(title='Grade', loc='upper left')

    # --- Graph 4: Previous GPA vs Attendance (Scatter Plot) ---
    for grade, group in data.groupby('FinalGrade'):
        axes[1, 1].scatter(
            group['PreviousGPA'], group['Attendance'],
            label=f'Grade {grade}',
            color=grade_colors.get(grade, 'gray'),
            s=80, alpha=0.8, edgecolors='black', linewidths=0.5
        )
    axes[1, 1].set_title('Previous GPA vs Attendance', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Previous GPA')
    axes[1, 1].set_ylabel('Attendance (%)')
    axes[1, 1].legend(title='Grade', loc='upper left')

    plt.tight_layout()
    save_path = f"{output_folder}/data_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Graphs saved to: {save_path}")


# ============================================================
# STEP 6: PREPARE DATA FOR ML MODEL
# ============================================================
def prepare_features(data):
    """
    Separates the data into:
      - X: Input features (what the model uses to learn)
      - y: Target/Output (what the model needs to predict)

    Also converts grade letters (A, B, C...) into numbers
    because ML models work with numbers, not letters.
    """
    print("\n" + "="*60)
    print("  PREPARING DATA FOR MACHINE LEARNING")
    print("="*60)

    # X = Features (input columns used for prediction)
    feature_columns = ['Attendance', 'StudyHours', 'AssignmentScore', 'PreviousGPA']
    X = data[feature_columns]

    # y = Target (the column we want to predict)
    y = data['FinalGrade']

    # LabelEncoder converts: A→0, B→1, C→2, D→3, F→4 (alphabetically)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n✅ Features (X) shape: {X.shape}  → {X.shape[0]} students, {X.shape[1]} features")
    print(f"✅ Target (y) shape:   {y_encoded.shape}")
    print(f"\n📝 Grade encoding used by the model:")
    for i, grade in enumerate(label_encoder.classes_):
        print(f"   {grade} → {i}")

    return X, y_encoded, label_encoder, feature_columns


# ============================================================
# STEP 7: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================
def split_data(X, y):
    """
    We split the data into two parts:
      - Training set (80%): The model learns from this
      - Testing set  (20%): We use this to check how well it learned

    This is like studying from a textbook and then taking a test
    with questions you haven't seen before.
    """
    # test_size=0.2 means 20% for testing, 80% for training
    # random_state=42 ensures we get the same split every time (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n" + "="*60)
    print("  TRAIN-TEST SPLIT")
    print("="*60)
    print(f"\n📚 Training samples: {len(X_train)} students  (80%)")
    print(f"🧪 Testing samples:  {len(X_test)} students  (20%)")

    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 8: TRAIN THE DECISION TREE MODEL
# ============================================================
def train_model(X_train, y_train):
    """
    Here we train (teach) the Decision Tree model.

    Decision Tree works like:
    IF Attendance > 85 AND StudyHours > 4 → Likely Grade A
    ELSE IF Attendance > 70 → Likely Grade B
    ...and so on (like a flowchart).

    max_depth=5 limits the tree so it doesn't overfit
    (memorize training data instead of learning patterns).
    """
    print("\n" + "="*60)
    print("  TRAINING THE MODEL")
    print("="*60)

    # Create the Decision Tree model
    model = DecisionTreeClassifier(
        max_depth=5,        # How deep the tree can grow (prevents overfitting)
        random_state=42,    # For reproducibility
        min_samples_leaf=2  # Each leaf must have at least 2 samples
    )

    # Train the model using the training data
    model.fit(X_train, y_train)

    print("\n✅ Model training complete!")
    print(f"   Algorithm: Decision Tree Classifier")
    print(f"   Max Depth: {model.max_depth}")
    print(f"   Number of leaves: {model.get_n_leaves()}")
    print(f"   Tree depth: {model.get_depth()}")

    return model


# ============================================================
# STEP 9: EVALUATE THE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test, label_encoder, output_folder="outputs"):
    """
    We check how accurate our model is on data it has never seen.
    This tells us how well it will perform in the real world.
    """
    print("\n" + "="*60)
    print("  MODEL EVALUATION")
    print("="*60)

    # Make predictions on test data
    y_predicted = model.predict(X_test)

    # Calculate accuracy (what % of predictions were correct)
    accuracy = accuracy_score(y_test, y_predicted)

    print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
    print(f"   (The model correctly predicted {int(accuracy * len(y_test))} out of {len(y_test)} test students)\n")

    # Detailed report: precision, recall, f1-score per grade
    print("📊 Detailed Classification Report:")
    print("-" * 50)
    print(classification_report(
        y_test, y_predicted,
        target_names=label_encoder.classes_
    ))

    # Feature Importance: Which feature helped the model most?
    print("🔑 Feature Importance (which factors matter most):")
    print("-" * 50)
    feature_names = ['Attendance', 'StudyHours', 'AssignmentScore', 'PreviousGPA']
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(importance * 40)
        print(f"   {name:<20} {bar} ({importance * 100:.1f}%)")

    # Save accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Attendance', 'Study Hours', 'Assignment Score', 'Previous GPA']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.barh(names, importances * 100, color=colors, edgecolor='black')
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title('Feature Importance in Grade Prediction', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, importances * 100):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontweight='bold')
    ax.set_xlim(0, max(importances * 100) + 12)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/feature_importance.png", dpi=150)
    plt.close()
    print(f"\n✅ Feature importance chart saved to: {output_folder}/feature_importance.png")

    return accuracy


# ============================================================
# STEP 10: PREDICT FOR A NEW STUDENT
# ============================================================
def predict_student(model, label_encoder):
    """
    This function lets you enter a new student's data manually
    and get a grade prediction from the trained model.
    """
    print("\n" + "="*60)
    print("  PREDICT GRADE FOR A NEW STUDENT")
    print("="*60)
    print("\nEnter the student's information below:")
    print("(Press Enter to use default example values)\n")

    try:
        attendance_input = input("📅 Attendance (%) [e.g., 85]: ").strip()
        attendance = float(attendance_input) if attendance_input else 85.0

        study_input = input("📖 Study Hours per Day [e.g., 5]: ").strip()
        study_hours = float(study_input) if study_input else 5.0

        assignment_input = input("📝 Assignment Score (0-100) [e.g., 78]: ").strip()
        assignment_score = float(assignment_input) if assignment_input else 78.0

        gpa_input = input("🎓 Previous GPA (0-10) [e.g., 7.5]: ").strip()
        previous_gpa = float(gpa_input) if gpa_input else 7.5

    except ValueError:
        print("⚠️  Invalid input detected. Using default values.")
        attendance, study_hours, assignment_score, previous_gpa = 85.0, 5.0, 78.0, 7.5

    # Create a single-row DataFrame with the student's data
    new_student = pd.DataFrame({
        'Attendance':       [attendance],
        'StudyHours':       [study_hours],
        'AssignmentScore':  [assignment_score],
        'PreviousGPA':      [previous_gpa]
    })

    # Predict the grade
    predicted_encoded = model.predict(new_student)
    predicted_grade = label_encoder.inverse_transform(predicted_encoded)[0]

    # Get probability scores for each grade
    probabilities = model.predict_proba(new_student)[0]

    print("\n" + "="*60)
    print("  PREDICTION RESULT")
    print("="*60)
    print(f"\n  Student Profile:")
    print(f"  ├─ Attendance:        {attendance}%")
    print(f"  ├─ Study Hours/Day:   {study_hours} hrs")
    print(f"  ├─ Assignment Score:  {assignment_score}/100")
    print(f"  └─ Previous GPA:      {previous_gpa}/10")
    print(f"\n  🏆 Predicted Final Grade: [ {predicted_grade} ]")

    print(f"\n  Grade Probabilities:")
    for grade, prob in zip(label_encoder.classes_, probabilities):
        bar = "█" * int(prob * 30)
        print(f"  {grade}: {bar} {prob*100:.1f}%")

    # Give advice based on predicted grade
    advice = {
        'A': "🌟 Excellent! Keep up the outstanding work!",
        'B': "👍 Good performance! A little more effort can get you an A.",
        'C': "⚠️  Average. Try to increase study hours and attendance.",
        'D': "🚨 Below average. Seek help from teachers immediately.",
        'F': "❌ At risk of failing. Please attend extra classes and revise thoroughly."
    }
    print(f"\n  💬 Advice: {advice.get(predicted_grade, 'Keep working hard!')}")
    print("\n" + "="*60)

    return predicted_grade


# ============================================================
# STEP 11: MAIN FUNCTION — RUNS EVERYTHING IN ORDER
# ============================================================
def main():
    """
    This is the entry point of the program.
    It calls all functions in the correct sequence.
    """
    import os
    os.makedirs("outputs", exist_ok=True)

    # Path to our dataset
    DATA_PATH = "data/students.csv"

    # --- Run each step in order ---
    data             = load_data(DATA_PATH)
    data             = clean_data(data)
    explore_data(data)
    visualize_data(data)
    X, y, encoder, features = prepare_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model            = train_model(X_train, y_train)
    accuracy         = evaluate_model(model, X_test, y_test, encoder)
    predict_student(model, encoder)

    print("\n✅ Project completed successfully!")
    print(f"   Final Model Accuracy: {accuracy*100:.2f}%")
    print("="*60 + "\n")


# This ensures main() runs only when we execute this file directly
if __name__ == "__main__":
    main()
