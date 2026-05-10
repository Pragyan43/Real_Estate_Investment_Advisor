import pandas as pd

# Load dataset
df = pd.read_csv("data/india_housing_prices.csv")

# Remove duplicates
df = df.drop_duplicates()

# Create Price per SqFt
df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

# Create Age of Property
df["Age_of_Property"] = 2026 - df["Year_Built"]

# Remove missing values
df = df.dropna()

# -----------------------------
# STEP 6 → Create Future Price
# -----------------------------

growth_rate = 0.08

df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"] * ((1 + growth_rate) ** 5)
)

# -----------------------------
# STEP 7 → Create Good Investment
# -----------------------------

median_price_per_sqft = df["Price_per_SqFt"].median()

df["Good_Investment"] = df["Price_per_SqFt"].apply(
    lambda x: 1 if x <= median_price_per_sqft else 0
)

# Save cleaned dataset
df.to_csv("data/cleaned_real_estate.csv", index=False)

print(df.head())

print("Preprocessing + Feature Engineering Completed!")