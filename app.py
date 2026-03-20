import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title("🚗 MPG Prediction")

# Upload dataset
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write(df.head())

    # Features and target
    X = df[['Displacement', 'Horsepower', 'Weight', 'Acceleration']]
    y = df['MPG']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Polynomial features
    poly = PolynomialFeatures(degree=2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual MPG")
    ax.set_ylabel("Predicted MPG")
    st.pyplot(fig)

    # Input for prediction
    st.write("### Predict MPG")

    d = st.number_input("Displacement")
    h = st.number_input("Horsepower")
    w = st.number_input("Weight")
    a = st.number_input("Acceleration")

    if st.button("Predict"):
        data = np.array([[d, h, w, a]])
        data = poly.transform(data)
        result = model.predict(data)
        st.success(f"MPG: {result[0]:.2f}")

else:
    st.write("Upload dataset to start")
