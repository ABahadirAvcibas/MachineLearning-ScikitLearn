import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/ABahadirAvcibas/ExamScore-Prediction-MachineLearning/refs/heads/main/StudentPerformanceFactors.csv"
rawData = pd.read_csv(url)
rawData.head(10)
rawData.tail()
rawData.isna().any()
rawData.describe()
rawData["Internet_Access"].value_counts()
rawData.isnull().sum()


def plotRelation(data, col1, col2):
  x=data[col1]
  y=data[col2]
  plt.figure(figsize=(7,4))
  plt.scatter(x, y)
  plt.xlabel(col1)
  plt.ylabel(col2)
  plt.title(f"{col1} vs {col2}")
  plt.tight_layout()
  plt.show()

rawData[["Attendance", "Exam_Score"]].corr()
plotRelation(rawData, "Sleep_Hours", "Exam_Score")
data = rawData.copy()
data["Teacher_Quality"] = data["Teacher_Quality"].fillna("Medium")
data["Parental_Education_Level"].value_counts()
data["Parental_Education_Level"] = data["Parental_Education_Level"].fillna("High School")
data["Distance_from_Home"] = data["Distance_from_Home"].fillna("Near")
data.isnull().sum()

# Ordinal Mapping
low_med_high = {"Low": 0, "Medium": 1, "High": 2}
dist_map = {"Near": 0, "Moderate": 1, "Far": 2}
edu_map = {"High School": 0, "College": 1, "Postgraduate": 2}
peer_map = {"Negative": 0, "Neutral": 1, "Positive": 2}

for c in ["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "Teacher_Quality"]:
    data[c] = data[c].map(low_med_high)

data["Distance_from_Home"] = data["Distance_from_Home"].map(dist_map)
data["Parental_Education_Level"] = data["Parental_Education_Level"].map(edu_map)
data["Peer_Influence"] = data["Peer_Influence"].map(peer_map)

nominal_cols = ["Extracurricular_Activities", "Internet_Access", "School_Type", "Learning_Disabilities", "Gender"]

data = pd.get_dummies(data, columns=nominal_cols, drop_first=True, dtype=int)

# Artık bütün veriler rakam

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

x = data.drop(columns=["Exam_Score"])
y = data["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=94)

lin_model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

# Train
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.3f}")

plt.figure(figsize=(7,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Prediction")
plt.tight_layout()
plt.show()