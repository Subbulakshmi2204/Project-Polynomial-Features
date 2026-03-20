import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title("🚗 MPG Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write("Columns in dataset:", df.columns)

    # Required columns
    required = ['displacement', 'horsepower', 'weight', 'acceleration', 'mpg']

    # Check if columns exist
    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Continue if no error
    X = df[['displacement', 'horsepower', 'weight', 'acceleration']]
    y = df['mpg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    poly = PolynomialFeatures(degree=2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    st.pyplot(fig)

    st.write("### Predict MPG")

    d = st.number_input("displacement")
    h = st.number_input("horsepower")
    w = st.number_input("weight")
    a = st.number_input("acceleration")

    if st.button("Predict"):
        data = np.array([[d, h, w, a]])
        data = poly.transform(data)
        result = model.predict(data)
        st.success(f"MPG: {result[0]:.2f}")

else:
    st.write("Upload dataset")
