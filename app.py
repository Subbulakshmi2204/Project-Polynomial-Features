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

    # Fix column names
    df.columns = df.columns.str.lower()

    st.write("Columns:", df.columns)

    # Select correct columns
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
