import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Price Prediction")
st.write("Machine Learning model using Linear Regression")


# =========================================================
# 2. LOAD DATASET
# =========================================================



# Main.py jahan saved hai wahi se path pick karega
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
# Yahan file ka naam Dataset.csv kar dijiye
csv_path = os.path.join(base_dir, "Dataset.csv")

df = pd.read_csv(csv_path)


# =========================================================
# 3. CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()


# =========================================================
# 4. SHOW DATASET INFORMATION
# =========================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Cars", len(df))

with col2:
    st.metric("Companies", df["brand"].nunique())

with col3:
    st.metric("Models", df["model_name"].nunique())


# =========================================================
# 5. SHOW COMPANY CATEGORIES
# =========================================================

st.subheader("🏢 Car Companies")

companies = sorted(df["brand"].dropna().unique())

st.write(", ".join(companies))


# =========================================================
# 6. SHOW FUEL CATEGORIES
# =========================================================

st.subheader("⛽ Fuel Categories")

fuel_types = sorted(df["fuel_type"].dropna().unique())

st.write(", ".join(fuel_types))


# =========================================================
# 7. SHOW CITY CATEGORIES
# =========================================================

st.subheader("📍 Cities")

cities = sorted(df["city"].dropna().unique())

st.write(", ".join(cities))


# =========================================================
# 8. PREPARE DATA FOR ML
# =========================================================

features = [
    "year",
    "brand",
    "model_name",
    "distance_travelled(kms)",
    "fuel_type",
    "city"
]

target = "price"


# Remove rows having missing values in selected columns

model_df = df[features + [target]].dropna()


X = model_df[features]

y = model_df[target]


# =========================================================
# 9. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 10. CATEGORICAL COLUMNS
# =========================================================

categorical_features = [
    "brand",
    "model_name",
    "fuel_type",
    "city"
]


numeric_features = [
    "year",
    "distance_travelled(kms)"
]


# =========================================================
# 11. ONE-HOT ENCODING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 12. LINEAR REGRESSION MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# =========================================================
# 13. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# 14. MODEL TESTING
# =========================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)


st.subheader("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "R² Score",
        round(r2, 3)
    )

with col2:
    st.metric(
        "Mean Absolute Error",
        f"₹{mae:,.0f}"
    )


# =========================================================
# 15. PRICE PREDICTION PAGE
# =========================================================

st.subheader("🔮 Predict Car Price")

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Company
# ---------------------------------------------------------

with col1:

    selected_brand = st.selectbox(
        "Select Company",
        companies
    )


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

brand_models = sorted(
    df[
        df["brand"] == selected_brand
    ]["model_name"]
    .dropna()
    .unique()
)

with col2:

    selected_model = st.selectbox(
        "Select Model",
        brand_models
    )


# ---------------------------------------------------------
# Year
# ---------------------------------------------------------

selected_year = st.number_input(
    "Car Manufacturing Year",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=int(df["year"].max())
)


# ---------------------------------------------------------
# Kilometres
# ---------------------------------------------------------

selected_km = st.number_input(
    "Kilometres Driven",
    min_value=0.0,
    value=30000.0
)


# ---------------------------------------------------------
# Fuel
# ---------------------------------------------------------

selected_fuel = st.selectbox(
    "Fuel Type",
    fuel_types
)


# ---------------------------------------------------------
# City
# ---------------------------------------------------------

selected_city = st.selectbox(
    "City",
    cities
)


# =========================================================
# 16. PREDICT BUTTON
# =========================================================

if st.button("🚀 Predict Car Price"):

    input_data = pd.DataFrame(
        {
            "year": [selected_year],
            "brand": [selected_brand],
            "model_name": [selected_model],
            "distance_travelled(kms)": [selected_km],
            "fuel_type": [selected_fuel],
            "city": [selected_city]
        }
    )

    predicted_price = model.predict(input_data)[0]

    st.success(
        f"💰 Estimated Car Price: ₹{predicted_price:,.0f}"
    )