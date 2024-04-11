import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time


model = joblib.load('model.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')



st.title("Titanic Survival Prediction")

st.image("img.jpeg",width=700)

st.write("Please enter all the details without leaving anything blank")

Age = st.slider("Enter your age")
st.write("Your age is :",Age)

SibSp = st.number_input("Number of siblings / spouses aboard the Titanic",min_value=0)

Parch = st.number_input("Number of parents / children aboard the Titanic",min_value=0)

Fare = st.number_input("How much did you pay for the ticket (in $)",min_value=0.0,step=1.0)

deck = st.slider("Enter your deck",min_value=1,max_value=8)

Pclass = st.selectbox('Please select the class: 1: Upper , 2: Middle , 3: Lower',[1,2,3])

Sex = st.selectbox('Please enter your gender',['male','female'])

Embarked = st.selectbox('Please enter the port on which you embarked C: Cherbourg, Q: Queenstown, S: Southampton',['C','Q','S'])

Titles = st.selectbox("Please enter your title",['Mr','Miss','Mrs','Master','Royalty','Officer'])



input_data = pd.DataFrame({'Age': [Age], 'SibSp': [SibSp], 'Parch': [Parch], 'Fare': [Fare], 'deck': [deck],'Pclass': [Pclass], 'Sex': [Sex], 'Embarked': [Embarked], 'Titles': [Titles]})
cols_test = ['Pclass','Sex','Embarked','Titles']
encoded_cols_test = encoder.transform(input_data[cols_test])

encoded_df_test = pd.DataFrame(encoded_cols_test, columns=encoder.get_feature_names_out(cols_test))

input_data = input_data.drop(columns=cols_test)

encoded_data_test = pd.concat([input_data, encoded_df_test], axis=1)


if st.button("Submit"):

    # Scalling the features
    
    X = encoded_data_test.iloc[:,:]
    
    X = scaler.transform(X)
    
    X_flat = np.ravel(X).tolist()
    
    X= X.reshape(1,15)
    
    predic = model.predict(X)
    
    progress = st.progress(0)
    for i in range (100):
        time.sleep(0.01)
        progress.progress(i+1)
        
    if predic == 0:
        st.info("You would not have survived the titanic")
        gif_url = 'shinking.gif'  
        st.image(gif_url, caption='Your GIF Caption', use_column_width=True)
    elif predic == 1:
        st.info("You would have survived the Titanic")
        st.balloons()
        
        
