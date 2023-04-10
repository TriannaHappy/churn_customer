import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page : ', ('EDA','Predict The Customer'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()