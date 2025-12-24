import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from numpy import loadtxt

# -------------------------------
# Load dataset and train model
# -------------------------------
dataset = loadtxt('data.csv', delimiter=',')

X = dataset[:, 0:8]
y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(8,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 Diabetes Risk Predictor -- Pima Native American Tribe")

st.write("Data is based off the overall population health of the Pima Native American tribe.")
st.write("Enter your health information below:")

pregnancies = st.number_input("Number of pregnancies", min_value=0, value=0)
glucose = st.number_input("Glucose level", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70)
skin_thickness = st.number_input("Skin thickness (mm)", min_value=0, value=20)
insulin = st.number_input("Insulin level (mu U/ml)", min_value=0, value=80)
bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
age = st.number_input("Age (years)", min_value=0, value=30)

# Button to predict
if st.button("Predict"):
    user_values = [pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, dpf, age]
    
    user_array = np.array([user_values])
    probability = model.predict(user_array, verbose=0)[0][0]
    
    if probability >= 0.7:
        risk = "High Risk ⚠️"
    elif probability >= 0.5:
        risk = "Moderate Risk ⚡"
    else:
        risk = "Low Risk ✅"
    
    st.subheader("📊 Prediction Results")
    st.write(f"**Diabetes Risk:** {probability*100:.2f}%")
    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Model Test Accuracy:** {test_accuracy*100:.2f}%")