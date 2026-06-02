import streamlit as st
import pickle
import numpy as np

# Load saved model with full path
model_path = "titanic_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Verify model loaded correctly
st.write(f"Model type: {type(model)}")

# App title
st.title('🚢 Titanic Survival Predictor')
st.write('Enter passenger details below to predict survival!')

# Input widgets
pclass = st.selectbox('Passenger Class', [1, 2, 3],
                       format_func=lambda x: f'Class {x}')
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', 1, 80, 25)
sibsp = st.slider('Siblings/Spouses on board', 0, 8, 0)
parch = st.slider('Parents/Children on board', 0, 6, 0)
fare = st.slider('Fare Paid ($)', 0, 500, 50)
embarked = st.selectbox('Port of Embarkation',
                         ['Southampton', 'Cherbourg', 'Queenstown'])
alone = st.selectbox('Travelling Alone?', ['Yes', 'No'])

# Convert inputs to numbers
sex_num = 0 if sex == 'Male' else 1
embarked_num = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]
alone_num = 1 if alone == 'Yes' else 0

# Predict button
if st.button('🔮 Predict Survival'):
    features = np.array([[pclass, sex_num, age,
                          sibsp, parch, fare,
                          embarked_num, alone_num]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.write("---")

    if prediction == 1:
        st.success('✅ This passenger would have SURVIVED!')
        st.info(f'Survival probability: {probability[1]*100:.1f}%')
    else:
        st.error('❌ This passenger would NOT have survived.')
        st.info(f'Survival probability: {probability[1]*100:.1f}%')

st.write("---")
st.write("*Model: Random Forest Classifier — 83.24% accuracy*")
