# Load the Models
import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model

# Read the Files

model_ann=load_model('churn_model.tf', compile = False)

with open('imputer_num.pkl', 'rb') as file_1:
    imputer_num=pickle.load(file_1)
    
with open('imputer_cat.pkl', 'rb') as file_m1:
    imputer_cat=pickle.load(file_m1)
    
with open('windsoriser.pkl', 'rb') as file_4:
    windsoriser=pickle.load(file_4)
    
with open('preprocessor.pkl', 'rb') as file_9:
    preprocessor=pickle.load(file_9)  

with open('num_cols.txt', 'r') as file_5:
    num_cols=json.load(file_5)

with open('cat_cols.txt','r') as file_6: 
    cat_cols=json.load(file_6)


def run():
    with st.form(key='form_customer_profile'):
        user_id = st.text_input('user_id', help='user id of the customer')
        age = st.number_input('Age',min_value=5, max_value=70, value=52, help='age of the customer')
        gender = st.selectbox('Gender', ('F', 'M'), index=0, help='gender of customer, F=Female, M=Male')
        region_category = st.selectbox('region', ('City', 'Village', 'Town'), index=0, help='region where customer live')
        membership_category = st.selectbox('Membership Category', ('No Membership', 'Basic Membership', 'Silver Membership',
        'Gold Membership', 'Platinum Membership','Premium Membership'), index=0, help='membership category class')
        
        st.markdown('---')
        joining_date = st.text_input('Joining Date', help='date when a customer became a member, format (YYYY-MM-DD)')
        joined_through_referral = st.selectbox('Joined through referral', ('No', 'Yes'), index=1, help='whether a customer joined using any referral code or ID')
        avg_time_spent = st.number_input('Average Time Spent', min_value=0., max_value=3500., value=301.67, help='average time spent by a customer on the website')
        avg_transaction_value = st.number_input('Average Transaction Value', min_value=400., max_value=110000.0, value=14545.56, help='average transaction value of a customer')
        avg_frequency_login_days = st.number_input('Average Frequency Login Days', min_value=0., max_value=80., value=16., help='number of times a customer has logged in to the website')
        last_visit_time = st.text_input('Last Visit Time', help='last time a customer visited the website, format (HH:MM::SS)')
        days_since_last_login = st.number_input('Days Since Last Login', min_value=0, max_value=30, value=11, help=' number of days since a customer last logged into he website')
        medium_of_operation = st.selectbox('Medium of Operation', ('Desktop', 'Smartphone', 'Both'), index=1, help='medium of operation that a customer uses for transactions')
        internet_option = st.selectbox('Internet Option', ('Wi-Fi', 'Fiber_Optic', 'Mobile_Data'), index=2, help='type of internet service customer uses')
        

        st.markdown('---')
        preferred_offer_types = st.selectbox('Preferred Offer Types', ('Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'), index=1, help='type of offer that a customer prefers')
        points_in_wallet = st.number_input('Points in Wallet', min_value=0., max_value=2500., value=0., help='points awarded to a customer on each transaction')
        used_special_discount = st.selectbox('Used Special Discount', ('No', 'Yes'), index=0, help='whether a customer uses special discounts offered')
        offer_application_preference = st.selectbox('Offer Application Preference', ('No', 'Yes'), index=1, help='whether a customers prefers offers')

        st.markdown('---')
        past_complaint = st.selectbox('Past Complaint', ('No', 'Yes'), index=1, help='whether a customer has raised any complaints')
        complaint_status = st.selectbox('Complaint Status', ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved','Solved in Follow-up'), index=3, help='whether a complaints raised by a customer was resolved')
        feedback = st.selectbox('Feedback', ('Poor Website', 'Poor Customer Service', 'Too many ads','Poor Product Quality', 'No reason specified',
                                              'Products always in Stock','Reasonable Price', 'Quality Customer Care', 'User Friendly Website'), index=2, help='feedback provided by a customer')
        st.markdown('---')
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'user_id' : user_id,
        'age' : age,
        'gender' : gender,
        'region_category' : region_category,
        'membership_category' : membership_category,
        'joining_date' : joining_date,
        'joined_through_referral' : joined_through_referral,
        'preferred_offer_types' :preferred_offer_types,
        'medium_of_operation':medium_of_operation,
        'internet_option':internet_option,
        'last_visit_time':last_visit_time,
        'days_since_last_login':days_since_last_login,
        'avg_time_spent':avg_time_spent,
        'avg_transaction_value':avg_transaction_value,
        'avg_frequency_login_days':avg_frequency_login_days,
        'points_in_wallet':points_in_wallet,
        'used_special_discount':used_special_discount,
        'offer_application_preference':offer_application_preference,
        'past_complaint':past_complaint,
        'complaint_status':complaint_status,
        'feedback':feedback

    }
    df=pd.DataFrame([data_inf])
    st.dataframe(df)

    if submitted:
        # Split X data based on data type
        X_test_num = df[num_cols]
        X_test_cat = df[cat_cols]

        # Handling Missing Value
        X_test_num=pd.DataFrame(imputer_num.transform(X_test_num), columns=num_cols)
        X_test_cat=pd.DataFrame(imputer_cat.transform(X_test_cat), columns=cat_cols)

        # Handling Outlier
        X_test_num_capped = windsoriser.transform(X_test_num)

        # Feature Selection
        X_test_num_capped.drop(['age'], axis=1, inplace=True)
        X_test_cat.drop(['past_complaint', 'complaint_status', 'last_visit_time', 'user_id'], axis=1, inplace=True)

        # Concat all X data
        X_test_concat=pd.concat([X_test_num_capped.reset_index(drop=True),X_test_cat.reset_index(drop=True)], axis=1)

        # Transform X data using column transformer pipeline (scaling and encoding)
        X_test_final=preprocessor.transform(X_test_concat)

        # Model Evaluation of test-set

        y_pred_seq2 = model_ann.predict(X_test_final)
        y_pred_seq2 = np.where(y_pred_seq2 >= 0.5, 1, 0)
        if y_pred_seq2==1:
            st.write('# The customer will be churning')
        else:
            st.write('# The customer will not be churning')

        
if __name__ == '__main__':
    run()
