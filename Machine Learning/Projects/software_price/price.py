import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("software_prices_large.csv")

# Define mappings for categorical values
category_mapping = {cat: idx for idx, cat in enumerate(df["Category"].unique())}
platform_mapping = {plat: idx for idx, plat in enumerate(df["Platform"].unique())}
subscription_mapping = {sub: idx for idx, sub in enumerate(df["Subscription Type"].unique())}

# Convert categorical columns to numerical
df["Category"] = df["Category"].map(category_mapping)
df["Platform"] = df["Platform"].map(platform_mapping)
df["Subscription Type"] = df["Subscription Type"].map(subscription_mapping)

# Selecting features and target variable
X = df.drop(columns=["Software Name", "Final Price"])  # Excluding name and target variable
y = df["Final Price"]

# Preprocessing pipeline (only scaling now)
preprocessor = StandardScaler()

# Model pipeline
model = Pipeline(steps=[
    ("scaler", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

def predict_price():
    """Get numerical user input and predict software price."""
    
    print("\nEnter software details for price prediction:")
    
    print("\nCategory Mapping:", category_mapping)
    category = int(input("Category (Enter corresponding number): ").strip())

    print("\nPlatform Mapping:", platform_mapping)
    platform = int(input("Platform (Enter corresponding number): ").strip())

    print("\nSubscription Mapping:", subscription_mapping)
    subscription = int(input("Subscription Type (Enter corresponding number): ").strip())

    features = int(input("Number of Features: ").strip())
    user_ratings = float(input("User Ratings (1.0 - 5.0): ").strip())
    company_reputation = int(input("Company Reputation (1 - 10): ").strip())
    release_year = int(input("Release Year (e.g., 2020): ").strip())
    market_demand = int(input("Market Demand Index (0 - 100): ").strip())
    competitor_pricing = int(input("Competitor Pricing: ").strip())

    # Create a DataFrame for user input
    input_data = pd.DataFrame([[category, platform, subscription, features, user_ratings, 
                                company_reputation, release_year, market_demand, competitor_pricing]], 
                              columns=X.columns)

    # Scale input using StandardScaler
    transformed_input = preprocessor.transform(input_data)

    # Predict price
    predicted_price = model.named_steps["regressor"].predict(transformed_input)[0]
    
    print(f"\nPredicted Software Price: ${predicted_price:.2f}")

# Run the prediction function
predict_price()
