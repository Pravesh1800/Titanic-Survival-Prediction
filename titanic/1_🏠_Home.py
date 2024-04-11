import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('theme', 'light')

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="📈",
    )

st.sidebar.success("Please select a page")



st.markdown("<h1 style='text-align: center;'>Welcome to the Titanic Survival Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Explore our training data and view the insights.</p>", unsafe_allow_html=True)
st.write("")
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])  # Create columns to center the image
with col2:
    st.image('titanic/home.png', use_column_width=True)
st.write("")
st.write("")

st.markdown(""" #### Find out if you could have survived the Titanic or not """)
st.markdown("""
<p style='text-align: justify;'>
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, resulting in the deaths of more than 1,500 passengers and crew.
This Streamlit app allows you to explore data about the passengers aboard the Titanic and predict whether you would have survived the disaster based on various factors such as age, gender, and class.
To get started, input your information in the sidebar and click the 'Predict' button to see the outcome.
Remember, this is just a fun exercise based on historical data. The real-life outcome of such a tragedy is a solemn reminder of the fragility of life and the importance of safety measures.
</p>""",unsafe_allow_html=True)

st.markdown(" #### Methedology ")
st.write("""
The methodology employed in this app involves the following steps:

1. **Data Acquisition**: The dataset used in this analysis is obtained from a Kaggle competition on the Titanic disaster. It includes various features such as passenger information, ticket class, and survival status.

2. **Data Preprocessing**: Before training the Random Forest model, the dataset undergoes preprocessing steps such as handling missing values, encoding categorical variables, and feature scaling.

3. **Model Training**: We use the Random Forest algorithm to train a predictive model. Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes for classification problems.

4. **Model Evaluation**: The trained model is evaluated using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score to assess its performance in predicting survival outcomes.

5. **Deployment**: The trained model is deployed in a Streamlit web application, allowing users to input their information and obtain predictions on whether they would have survived the Titanic disaster.

This methodology ensures a systematic approach to data analysis and prediction, providing users with valuable insights into the factors influencing survival on the Titanic.
""")


st.write("")
st.write("")
st.markdown("<h3 style='text-align: center;'>Explore Our Analysis</h3>", unsafe_allow_html=True)
st.write("")
st.write("")
a1, a2, a3 = st.columns([1, 2, 1])  # Create columns to center the image
with a2:
    st.image('titanic/analysis.png', use_column_width=True)
st.write("")
st.write("")

# Analysis
st.write("**This is our training dataset**")
data = pd.read_csv('titanic/train.csv')
st.write(data)
st.write("_in survived column (1: Survived 2: Didn't Survived)_")

st.write("")
st.write("")
st.write("This chart shows the corelation between our features")
matrix = data.corr(numeric_only = True)
fig, ax = plt.subplots()
sns.heatmap(matrix, cmap='Greens') 
st.pyplot(fig)



