import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Used Car Price Prediction", layout="wide")

# =====================
# 1. Load Dataset
# =====================
st.title("🚗 Used Car Price Prediction Dashboard")
st.write("Analisis dataset mobil bekas dan pemodelan regresi untuk memprediksi harga jual.")

df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

st.subheader("📥 Data Preview")
st.dataframe(df.head())
st.write(f"Dataset memiliki **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")

# =====================
# 2. Data Cleaning
# =====================
st.header("🧹 Data Cleaning")

df_clean = df.copy()
df_clean = df_clean.drop_duplicates()

st.write("Jumlah data setelah menghapus duplikasi:", df_clean.shape[0])
st.write("Jumlah missing values per kolom:")
st.write(df_clean.isnull().sum())

# =====================
# 3. Data Manipulation
# =====================
st.header("🔧 Data Manipulation")

df_clean["brand"] = df_clean["name"].str.split().str[0]
df_clean = df_clean.drop(columns=["name"])

current_year = datetime.datetime.now().year
df_clean["car_age"] = current_year - df_clean["year"]
df_clean = df_clean.drop(columns=["year"])

st.dataframe(df_clean.head())

# =====================
# 4. EDA
# =====================
st.header("📊 Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Brand Distribution", "Car Age", "Selling Price"])

with tab1:
    brand_counts = df_clean["brand"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=brand_counts.index, y=brand_counts.values, ax=ax)
    ax.set_title("Top 10 Most Common Car Brands")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x="car_age", data=df_clean, order=sorted(df_clean["car_age"].unique()), ax=ax)
    ax.set_title("Distribution of Car Age")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_clean["selling_price"], bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of Selling Prices")
    st.pyplot(fig)

# =====================
# 5. Mapping Brand Tier
# =====================
st.header("🏷️ Mapping Brand Tier")

high_tier = ["Mercedes-Benz", "BMW", "Audi", "Jaguar", "Volvo"]
mid_tier = ["Toyota", "Honda", "Hyundai", "Skoda", "Mahindra", "Jeep"]
low_tier = ["Maruti", "Tata", "Renault", "Datsun", "Ford", "Nissan", "Chevrolet"]

def brand_tier(brand):
    if brand in high_tier:
        return "High"
    elif brand in mid_tier:
        return "Mid"
    elif brand in low_tier:
        return "Low"
    else:
        return "Other"

df_clean["brand_tier"] = df_clean["brand"].apply(brand_tier)
df_clean = df_clean.drop(columns=["brand"])

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="brand_tier", data=df_clean, order=["High", "Mid", "Low", "Other"], ax=ax)
ax.set_title("Distribution of Brand Tiers")
st.pyplot(fig)

# =====================
# 6. Feature Engineering
# =====================
st.header("⚙️ Feature Engineering")

X = df_clean.drop(columns="selling_price")
y = df_clean["selling_price"]

brand_tier_map = {"Low": 0, "Mid": 1, "High": 2, "Other": 3}
X["brand_tier"] = X["brand_tier"].map(brand_tier_map)

categorical_cols = ["fuel", "seller_type", "transmission", "owner"]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True).astype(int)

st.dataframe(X.head())

# =====================
# 7. Split Train-Test
# =====================
st.header("✂️ Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
st.write("Train set:", X_train.shape, " | Test set:", X_test.shape)

# =====================
# 8. Multicollinearity Test
# =====================
st.header("🔍 Multicollinearity Test")

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data.sort_values(by="VIF", ascending=False)

st.dataframe(vif_data.head(10))

# =====================
# 9. Modeling
# =====================
st.header("🤖 Regression Modeling")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Dataset": "Train",
        "MAE": mean_absolute_error(y_train, y_train_pred),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "R2": r2_score(y_train, y_train_pred)
    })
    results.append({
        "Model": name,
        "Dataset": "Test",
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "R2": r2_score(y_test, y_test_pred)
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df)

best_model = results_df[results_df["Dataset"] == "Test"].sort_values("R2", ascending=False).iloc[0]
st.success(f"✅ Model terbaik adalah **{best_model['Model']}** dengan R² = {best_model['R2']:.3f}")

# =====================
# 10. Prediction Form
# =====================
st.header("📈 Predict Selling Price")

with st.form("predict_form"):
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    car_age = st.number_input("Car Age (Years)", min_value=0, value=5)
    brand_tier_input = st.selectbox("Brand Tier", ["Low", "Mid", "High", "Other"])
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame({
        "km_driven": [km_driven],
        "car_age": [car_age],
        "brand_tier": [brand_tier_map[brand_tier_input]],
        "fuel_Diesel": [1 if fuel == "Diesel" else 0],
        "fuel_LPG": [1 if fuel == "LPG" else 0],
        "fuel_Petrol": [1 if fuel == "Petrol" else 0],
        "seller_type_Individual": [1 if seller_type == "Individual" else 0],
        "seller_type_Trustmark Dealer": [1 if seller_type == "Trustmark Dealer" else 0],
        "transmission_Manual": [1 if transmission == "Manual" else 0],
        "owner_Second Owner": [1 if owner == "Second Owner" else 0],
        "owner_Third Owner": [1 if owner == "Third Owner" else 0]
    })

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    prediction = model.predict(input_df)[0]

    st.metric("Predicted Selling Price (INR)", f"{prediction:,.0f}")

