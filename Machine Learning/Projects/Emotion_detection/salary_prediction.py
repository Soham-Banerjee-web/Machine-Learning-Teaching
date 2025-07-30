import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("salary_data_cleaned.csv")

# Select features and target
features = [
    'Job Title', 'Job Description', 'Rating', 'Industry', 'Sector', 'Size',
    'Founded', 'python_yn', 'R_yn', 'spark', 'aws', 'excel'
]
target = 'avg_salary'

X = df[features]
y = df[target]

# Define preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(max_features=300), 'Job Description'),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), ['Job Title', 'Industry', 'Sector', 'Size']),
    ('num', 'passthrough', ['Rating', 'Founded', 'python_yn', 'R_yn', 'spark', 'aws', 'excel'])
])

# Create the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the model
joblib.dump(model_pipeline, "salary_predictor_model.pkl")
print("Model saved to salary_predictor_model.pkl")
