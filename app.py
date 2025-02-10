import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Load the dataset
def load_data():
    data_path = "data/train.csv"  # Ensure correct file name
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()
    df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    return df

df = load_data()

# Streamlit App Header
st.title("✈️ Airline Passenger Satisfaction Analysis")
st.write("Explore the factors influencing passenger satisfaction and predict satisfaction levels using machine learning.")

# Show Dataset Overview
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Exploratory Data Analysis
st.header("📊 Data Visualizations")
if st.checkbox("Show Satisfaction Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='satisfaction', hue='customer type', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Delay Impact on Satisfaction"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='departure delay in minutes', y='arrival delay in minutes', hue='satisfaction', palette='coolwarm', ax=ax)
    st.pyplot(fig)

# Machine Learning Model
st.header("🤖 Machine Learning Model: Random Forest Classifier")

# Prepare Data
X = df.drop(columns=['satisfaction', 'id', 'unnamed: 0'], errors='ignore')
X = pd.get_dummies(X)  # Convert categorical data
Y = df['satisfaction']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

st.write(f"**Model Accuracy:** {accuracy:.2f}")

# User Input Prediction
st.header("🎯 Predict Passenger Satisfaction")
st.write("Enter passenger details to predict satisfaction:")

# User Inputs
customer_type = st.selectbox("Customer Type", df['customer type'].unique())
travel_type = st.selectbox("Type of Travel", df['type of travel'].unique())
travel_class = st.selectbox("Class", df['class'].unique())
dep_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=500, value=0)
arr_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=500, value=0)

# Convert Input to Model Format
input_data = pd.DataFrame([[customer_type, travel_type, travel_class, dep_delay, arr_delay]], 
                           columns=['customer type', 'type of travel', 'class', 'departure delay in minutes', 'arrival delay in minutes'])
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Make Prediction
if st.button("Predict Satisfaction"):
    prediction = clf.predict(input_data)[0]
    result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"
    st.write(f"### 🎉 Prediction: The passenger is **{result}**")