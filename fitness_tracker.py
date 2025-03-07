import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# ----------- Load Configuration from config.yaml -----------
try:
    with open(r"C:\Users\BHOMIK KANDPAL\OneDrive\Desktop\fitnesstracker\config.yaml.txt", "r", encoding="utf-8") as file:

        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("⚠️ config.yaml not found! Please create the file in the project directory.")
    st.stop()

# ----------- Initialize Authentication -----------
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ----------- Login Section -----------
name, authentication_status, username = authenticator.login('Login', location='sidebar')


if authentication_status:
    st.sidebar.success(f"Welcome, {name}!")
    authenticator.logout("Logout", "sidebar")

    # ---------- Fitness Tracker WebApp -----------
    st.write("## Personal Fitness Tracker")
    st.write("Predict your calories burned based on input parameters.")

    st.sidebar.header("User Input Parameters: ")

    def user_input_features():
        age = st.sidebar.slider("Age: ", 10, 100, 30)
        bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
        duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
        heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
        body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
        gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

        gender = 1 if gender_button == "Male" else 0

        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            "Gender_male": gender
        }

        return pd.DataFrame(data_model, index=[0])

    df = user_input_features()

    st.write("---")
    st.header("Your Parameters: ")
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.write(df)

    # Load and preprocess data
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
    except FileNotFoundError:
        st.error("⚠️ Missing dataset files! Ensure calories.csv and exercise.csv exist.")
        st.stop()

    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    for data in [exercise_train_data, exercise_test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)

    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]

    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]

    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)

    df = df.reindex(columns=X_train.columns, fill_value=0)

    prediction = random_reg.predict(df)

    st.write("---")
    st.header("Prediction: ")
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

    st.write(f"🔥 You will burn approximately **{round(prediction[0], 2)} kilocalories**.")

    st.write("---")
    st.header("Similar Results: ")
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))

    st.write("---")
    st.header("General Insights: ")

    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

    st.write(f"📊 You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of people.")
    st.write(f"⏳ Your exercise duration is longer than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of people.")
    st.write(f"💓 Your heart rate is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of people.")
    st.write(f"🌡️ Your body temperature is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of people.")

elif authentication_status is False:
    st.error("❌ Incorrect username or password. Please try again.")

elif authentication_status is None:
    st.warning("🔑 Please enter your username and password.")
