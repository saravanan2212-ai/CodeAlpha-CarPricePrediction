import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("car data.csv")

print("First 5 rows:")
print(df.head())

print("\n--- Info ---")
print(df.info())

print("\n--- Missing values ---")
print(df.isnull().sum())

# 2. Drop rows where Selling_Price is missing
df = df.dropna(subset=['Selling_Price'])

# 3. Encode categorical columns
le = LabelEncoder()
cat_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 4. Define input (X) and output (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluation
print("\n--- Model Performance ---")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
