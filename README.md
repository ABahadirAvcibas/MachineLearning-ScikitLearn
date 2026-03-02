# Student Exam Performance(kaggle) — Score Prediction (Regression)

This project builds a regression model to predict **Exam_Score** using student-related features (sleep hours, attendance, resources, family context, etc.).  
It includes a full machine learning workflow: **EDA → preprocessing → encoding → model training → evaluation**.

## Dataset
- Rows: ~6.6k
- Target: `Exam_Score`
- Features include:
  - Numeric: `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`
  - Categorical/Ordinal: `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`, etc.

> Note: If the dataset license/source requires attribution, add the original dataset link here and avoid uploading the raw data to the repository.

## Project Workflow
### 1) Exploratory Data Analysis (EDA)
- Checked dataset structure, datatypes, and missing values
- Reviewed distributions and basic relationships (e.g., Hours_Studied vs Exam_Score)

### 2) Preprocessing
- Filled missing values (categorical) using the **most frequent** category (mode)
- Encoded **ordinal** variables with meaningful order (e.g., Low/Medium/High → 0/1/2)
- Applied **One-Hot Encoding** for nominal categorical features

### 3) Modeling
- Trained a baseline **Linear Regression** model using a `Pipeline`
- Applied `StandardScaler` to input features before training

### 4) Evaluation Metrics
Regression is evaluated using:
- **MAE (Mean Absolute Error)**: average absolute prediction error
- **RMSE (Root Mean Squared Error)**: penalizes larger errors more
- **R² (R-squared)**: how much variance the model explains

Example result (may vary by run):
- MAE ≈ 0.48  
- RMSE ≈ 2.03  
- R² ≈ 0.71
