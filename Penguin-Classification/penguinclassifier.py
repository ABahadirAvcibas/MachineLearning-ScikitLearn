import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/ABahadirAvcibas/MachineLearning-ScikitLearn/refs/heads/main/Penguin-Classification/penguins_size.csv'

rawData = pd.read_csv(url)
rawData["sex"] = rawData["sex"].replace(".", np.nan)

rawData["sex"] = rawData["sex"].fillna(rawData["sex"].mode()[0])  # Filling up the sex column with the mode value

from sklearn.model_selection import train_test_split

x = rawData.drop("species", axis=1)
y = rawData["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=94)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() # We are gonna apply label encoder to the 'sex' column  LABEL ENCODER = 0/1, ONE HOT ENCODER = 0,1,2

x_train["sex"] = le.fit_transform(x_train["sex"])
x_test["sex"] = le.transform(x_test["sex"])       # With those lines we just make sex either 0 or 1 for train and test variables

# culmen_length_mm	culmen_depth_mm	 flipper_length_mm	 body_mass_g - now we will fill the missing values for numerical datas

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

numerical_columns = ["culmen_length_mm", "culmen_depth_mm",	"flipper_length_mm"	,"body_mass_g"]

x_train[numerical_columns] = imputer.fit_transform(x_train[numerical_columns])

x_test[numerical_columns] = imputer.transform(x_test[numerical_columns])

x_test.isnull().sum() # Both x_test and x_train values are now filled

# Adalar (Biscoe, Dream, Torgersen) arasında "Biscoe, Dream'den daha büyüktür" gibi bir hiyerarşi olmadığı için, onları One-Hot Encoding ile her ada için ayrı bir sütun
# açarak kodlamak en sağlıklısıdır.
# Since no island has a priority among each other, we are gonna apply OHE

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

island_train_encoded = ohe.fit_transform(x_train[["island"]])
island_test_encoded = ohe.transform(x_test[["island"]])

column_names = ohe.get_feature_names_out(["island"])

island_train_df = pd.DataFrame(island_train_encoded, columns=column_names, index=x_train.index)
island_test_df = pd.DataFrame(island_test_encoded, columns=column_names, index=x_test.index)

x_train = pd.concat([x_train.drop('island', axis=1), island_train_df], axis=1)
x_test = pd.concat([x_test.drop('island', axis=1), island_test_df], axis=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

## KORELASYON ANALİZİ

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(random_state=94)

from sklearn.model_selection import cross_val_score

model.fit(x_train, y_train)

# CROSS VALIDATION
scores = cross_val_score(model, x_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

feature_importance = pd.DataFrame(model.coef_, columns=x_train if 'x_train_cols' in locals() else [f'Feature {i}' for i in range(x_train.shape[1])])
print("\nModel Coefficients (first class):\n", feature_importance.iloc[0])

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)