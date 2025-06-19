import pickle
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# # Part 1: Data
# Import data
csv_filename = 'C:\\Users\\User\\.cache\\kagglehub\\models\\gsaha123\\diabetes-risk-assessment\\scikitLearn\\2gd\\1\\diabetes.csv'
dataset = pd.read_csv(csv_filename)

# data cleaning
dataset_new = dataset
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)
# NA value
dataset_new.isnull().sum()
dataset_new["Glucose"] = dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean())
dataset_new["BloodPressure"] = dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean())
dataset_new["SkinThickness"] = dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean())
dataset_new["Insulin"] = dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean())
dataset_new["BMI"] = dataset_new["BMI"].fillna(dataset_new["BMI"].mean())

# Feature scaling & selecting
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_new)
dataset_scaled = pd.DataFrame(dataset_scaled)
X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values
# data splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'])


# # Part 2: KNN Model
# Model Training
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)

# model Testing & Accuracy
Y_pred_knn=knn.predict(X_test)
score_knn = round(accuracy_score(Y_test, Y_pred_knn)*100, 2)
# Save & Load the model
pickle.dump(knn, open('Diabetesmodel.pkl', 'wb'))
loaded_model = pickle.load(open('Diabetesmodel.pkl', 'rb'))


if st.button("Quit App Now"):
    st.warning("Shutting down... You may close this window.")
    time.sleep(1)
    os._exit(0)  # Immediately terminates Python process

# # Part 3: Predict
# Input data
st.title("Diabetes Risk Assessment")
Age = st.number_input("Age", min_value=0, max_value=120, step=1)
Gl = st.number_input("Glucose (fasting plasma glucose in mg/dL)", min_value=20.0, max_value=500.0, step=0.1)
In = st.number_input("Insulin (fasting insulin in µU/mL)", min_value=0.0, max_value=900.0, step=0.1)
enabled = st.toggle("Enable BMI Calculator")
if enabled:
    # --- Weight Input ---
    weight_unit = st.selectbox("Select weight unit:", ["kg", "lb", "st"])
    if weight_unit == "kg":
        weight = st.number_input("Enter your weight (kg):", min_value=0.0)
    elif weight_unit == "lb":
        weight_lb = st.number_input("Enter your weight (pounds):", min_value=0.0)
        weight = weight_lb * 0.453592  # Convert lb to kg
    elif weight_unit == "st":
        weight_st = st.number_input("Enter your weight (stones):", min_value=0.0)
        weight = weight_st * 6.35029  # Convert stones to kg

    # --- Height Input ---
    height_unit = st.selectbox("Select height unit:", ["cm", "ft & in"])
    if height_unit == "cm":
        height_cm = st.number_input("Enter your height (cm):", min_value=0.0)
        height_m = height_cm / 100  # Convert cm to meters
    elif height_unit == "ft & in":
        feet = st.number_input("Feet:", min_value=0)
        inches = st.number_input("Inches:", min_value=0.0)
        total_inches = (feet * 12) + inches
        height_m = total_inches * 0.0254  # Convert inches to meters

    # --- Calculate BMI ---
    if st.button("Calculate BMI"):
        if height_m > 0 and weight > 0:
            Bmi = round(weight / (height_m ** 2), 1)
            st.info(f"Your BMI is: {Bmi}")
        else:
            st.error("Height and Weight must be greater than zero.")
            Bmi = 0
        os.environ["Bmi"] = str(Bmi)
else:
    os.environ["Bmi"] = str(st.number_input("BMI (body mass index in kg/m²)", min_value=10.0, max_value=70.0, step=0.1))

# Make prediction
if st.button("Predict"):
    # Scale all features of the new input data using the *same* fitted scaler
    input_data_original_format = np.array([[0, Gl, 0, 0, In, float(os.getenv("Bmi")), 0, Age, 0]])
    scaled_all_features = sc.transform(input_data_original_format)
    input_scaled = scaled_all_features[:, [1, 4, 5, 7]]
    # Prediction
    prediction = loaded_model.predict(input_scaled)
    if prediction[0] == 0:
        st.success('The person is not diabetic')
    else:
        st.warning('The person is diabetic')
    time.sleep(1)
    st.info(
        f"The Diabetes Risk Assessment model is based on a publicly available model from Kaggle (By GSAHA123 · Created On 2024.07.10) and trained using the Diabetes dataset from the UCI Machine Learning Repository. The model has been modified to use the K-Nearest Neighbors (KNN) algorithm, achieving an accuracy of approximately {str(score_knn)}%.")
    st.info(
        "This tool is intended solely for informational and educational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. The model's predictions depend heavily on the accuracy and quality of the input data provided by the user. Environmental factors and data inconsistencies may impact the model’s performance and reliability.")
    st.info(
        "Important: This model should not be used for critical or clinical decision-making without consultation with qualified healthcare professionals. Always seek advice from a licensed medical provider for questions regarding a medical condition or health-related decisions.")