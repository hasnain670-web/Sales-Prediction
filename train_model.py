import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

from  sklearn.metrics import(
    mean_absolute_error,mean_squared_error,
    r2_score
)

# Load Data
df=pd.read_csv("datasets/advertising.csv")

print("\nFirst 5 Rows")
print("df.head()")

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# Handle Missing Values
df.fillna(df.mean(numeric_only=True),inplace=True)

# STATISTICS Summary
print("\nStatistics")
print(df.describe())

# Correlation
print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))

# Data Visualization
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

sns.pairplot(df)
plt.show()

# Feature Target
x=df[["TV","Radio","Newspaper"]]
y=df["Sales"]

# Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Train Model
model=LinearRegression()
model.fit(x_train,y_train)
print("\nModel Traning Completed")

#Prediction
y_pred=model.predict(x_test)

#Evaluation
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=mse ** 0.5
r2=r2_score(y_test,y_pred)

print("\nModel Evaluation")
print("--------------------")
print("MAE :",round(mae,2))
print("MSE :",round(mse,2))
print("RMSE :",round(rmse,2))
print("R2 :",round(r2,2))

#Save Model
os.makedirs("models",exist_ok=True)
joblib.dump('models/sales_model.pkl')
print("\nModel Saved Successfully!")
