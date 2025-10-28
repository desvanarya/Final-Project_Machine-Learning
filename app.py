# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/desvanarya/Final-Project_Machine-Learning/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("🚗 Prediksi Harga Mobil Bekas")
st.markdown("Aplikasi Streamlit untuk memprediksi harga mobil bekas berdasarkan dataset **Car Dekho**.")

# =========================
# DATA CLEANING
# =========================
df_clean = df.copy().drop_duplicates()

# Extract brand
df_clean['brand'] = df_clean['name'].str.split().str[0]
df_clean.drop(columns=['name'], inplace=True)

# Create car age
import datetime
current_year = datetime.datetime.now().year
df_clean['car_age'] = current_year - df_clean['year']
df_clean.drop(columns=['year'], inplace=True)

# =========================
# BRAND TIER MAPPING
# =========================
high_tier = ['Mercedes-Benz', 'BMW', 'Audi', 'Jaguar', 'Volvo']
mid_tier = ['Toyota', 'Honda', 'Hyundai', 'Skoda', 'Mahindra', 'Jeep']
low_tier = ['Maruti', 'Tata', 'Renault', 'Datsun', 'Ford', 'Nissan', 'Chevrolet']
other_tier = ['Land', 'MG', 'Daewoo', 'Force', 'Isuzu', 'OpelCorsa', 'Ambassador', 'Kia']

def brand_tier(brand):
    if brand in high_tier:
        return 'High'
    elif brand in mid_tier:
        return 'Mid'
    elif brand in low_tier:
        return 'Low'
    else:
        return 'Other'

df_clean['brand_tier'] = df_clean['brand'].apply(brand_tier)
df_clean.drop(columns=['brand'], inplace=True)

# =========================
# FEATURE ENCODING
# =========================
X = df_clean.drop(columns='selling_price')
y = df_clean['selling_price']

brand_tier_map = {'Low': 0, 'Mid': 1, 'High': 2, 'Other': 3}
X['brand_tier'] = X['brand_tier'].map(brand_tier_map)

categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model():
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

model = train_model()

# =========================
# EDA SECTION
# =========================
st.subheader("📊 Exploratory Data Analysis (EDA)")

if st.checkbox("Tampilkan heatmap korelasi"):
    corr = df_clean.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.checkbox("Tampilkan distribusi harga"):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df_clean['selling_price'], bins=40, kde=True, ax=ax)
    ax.set_title("Distribusi Harga Mobil")
    st.pyplot(fig)

# =========================
# PREDIKSI
# =========================
st.subheader("🔮 Prediksi Harga Mobil")

km_driven = st.number_input("Kilometer Ditempuh", min_value=0, value=30000)
car_age = st.number_input("Usia Mobil (Tahun)", min_value=0, value=5)
brand_tier_input = st.selectbox("Brand Tier", ["Low", "Mid", "High", "Other"])
fuel = st.selectbox("Tipe Bahan Bakar", df_clean['fuel'].unique())
seller_type = st.selectbox("Tipe Penjual", df_clean['seller_type'].unique())
transmission = st.selectbox("Transmisi", df_clean['transmission'].unique())
owner = st.selectbox("Kepemilikan", df_clean['owner'].unique())

submitted = st.button("Prediksi Harga")

if submitted:
    input_df = pd.DataFrame({
        "km_driven": [km_driven],
        "car_age": [car_age],
        "brand_tier": [brand_tier_map[brand_tier_input]],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner]
    })

    # One-hot encode input seperti X_train
    input_df = pd.get_dummies(input_df, columns=["fuel", "seller_type", "transmission", "owner"], drop_first=True).astype(int)

    # Pastikan semua kolom sama dengan X_train
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_train.columns]

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Prediksi Harga Mobil: **₹{prediction:,.0f}** (INR)")

# =========================
# EVALUASI MODEL
# =========================
st.subheader("📈 Evaluasi Model")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

results = pd.DataFrame([
    ["Train", mean_absolute_error(y_train, y_train_pred),
     mean_squared_error(y_train, y_train_pred),
     np.sqrt(mean_squared_error(y_train, y_train_pred)),
     r2_score(y_train, y_train_pred)],
    ["Test", mean_absolute_error(y_test, y_test_pred),
     mean_squared_error(y_test, y_test_pred),
     np.sqrt(mean_squared_error(y_test, y_test_pred)),
     r2_score(y_test, y_test_pred)]
],
columns=["Dataset", "MAE", "MSE", "RMSE", "R2"])

st.dataframe(results)
