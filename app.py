import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title("🚗 MPG Prediction (Polynomial Regression)")

file = st.file_uploader("Upload Auto MPG CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write("📌 Columns in Dataset:", df.columns)

    # Function to find column by keyword
    def find_col(keywords):
        for col in df.columns:
            for key in keywords:
                if key in col:
                    return col
        return None

    # Auto-detect columns
    displacement = find_col(["displacement"])
    horsepower = find_col(["horse", "hp"])
    weight = find_col(["weight"])
    acceleration = find_col(["acceleration", "acc"])
    mpg = find_col(["mpg", "miles"])

    # Check if all required columns found
    if None in [displacement, horsepower, weight, acceleration, mpg]:
        st.error("❌ Could not detect required columns.")
        st.write("👉 Please check your dataset column names above.")
        st.stop()

    # Prepare data
    X = df[[displacement, horsepower, weight, acceleration]]
    y = df[mpg]

    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Polynomial features
    poly = PolynomialFeatures(degree=2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Plot
    st.subheader("📉 Actual vs Predicted MPG")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual MPG")
    ax.set_ylabel("Predicted MPG")
    st.pyplot(fig)

    # User input
    st.subheader("🔮 Predict MPG")

    d = st.number_input("Displacement", value=200.0)
    h = st.number_input("Horsepower", value=100.0)
    w = st.number_input("Weight", value=3000.0)
    a = st.number_input("Acceleration", value=15.0)

    if st.button("Predict"):
        data = np.array([[d, h, w, a]])
        data = poly.transform(data)
        result = model.predict(data)
        st.success(f"🚘 Predicted MPG: {result[0]:.2f}")

else:
    st.info("Please upload dataset to continue.")
