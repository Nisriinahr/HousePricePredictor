import pandas as pd
import numpy as np

#dataset
file_path = r'C:\Users\nisriina hilmi\Downloads\AmesHousing.csv'
df = pd.read_csv(file_path)

#1000 data
df = df.head(1000)

selected_features = [
    'MS SubClass',
    'Lot Area',
    'Street',
    'Lot Config',
    'House Style',
    'Bldg Type',
    'Overall Qual',
    'Overall Cond',
    'Year Built',
    'Year Remod/Add',
    'Foundation',
    'TotRms AbvGrd',
    'Fireplaces',
    'Garage Area',
    'Yr Sold',
    'Sale Type',
    'Sale Condition',
    'SalePrice'
]

df = df[selected_features]
df = df.dropna()
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

output_path = r"C:\Users\nisriina hilmi\Downloads\preprocessed data.csv"
try:
    df.to_csv(output_path, index=False)
    print("File berhasil disimpan di:", output_path)
except PermissionError:
    print("Error: Tidak ada izin menulis di folder. Coba simpan di lokasi lain.")
print(f"Final dataset shape: {df.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X_encoded = pd.get_dummies(X, columns=[
    'Street',
    'Lot Config', 
    'House Style',
    'Bldg Type',
    'Foundation',
    'Sale Type',
    'Sale Condition'
])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
num_cols = ['MS SubClass', 'Lot Area', 'Overall Qual', 'Overall Cond', 
            'Year Built', 'Year Remod/Add', 'TotRms AbvGrd', 
            'Fireplaces', 'Garage Area', 'Yr Sold']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

coefficients = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print(model.coef_)
print(model.intercept_)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

import joblib
model_path = r"C:\Users\nisriina hilmi\Downloads\linear_regression_model.pkl"
try:
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully at: {model_path}")
except PermissionError:
    print("\nError: Could not save model. Check directory permissions.")
