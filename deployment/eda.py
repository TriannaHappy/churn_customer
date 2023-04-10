import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from PIL import Image

# Untuk melebarkan streamlit, harus diletakkan setelh import
# Ketika dieksekusi akan mempengaruhi main dan prediction
# Tidak perlu dijalankan dalam fungsi
st.set_page_config(
    page_title='Churn Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)


# bagian bawah ini tidak bisa dijalankan jika tidak dieksekusi
def run():
    #Membuat Title
    st.title('Churn Prediction Using Artifical Neural Network')

    # Membuat Sub Header
    st.subheader('EDA for Customer Dataset')

    # Menambahkan Gambar
    image=Image.open('Customer.jpg')
    st.markdown(
    """
    <style>
    img {
        cursor: pointer;
        transition: all .2s ease-in-out;
    }
    img:hover {
        transform: scale(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.image(image, caption='Churn Prediction')

    # Menambah Deskripsi
    st.write('Made by *Happy Trianna*')
    
    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    Pada page kali ini, pennulis akan melakukan eksplorasi sederhana.
    Dataset yang digunakan adalah dataset customer dari toko berbasis online.
    Dataset ini berasal dari web github.com.
    '''

    # Show DataFrame
    st.write('#### Head of Customer Profil Dataset')
    df = pd.read_csv('churn.csv')
    st.dataframe(df.head(10))

    st.write('#### Describe of The Customer Profil Dataset')
    st.dataframe(df.describe())

    #Check the distribution of clients, categorized by target who subsribed the term deposit
    st.write('#### Churn Risk Score Distribution')
    fig = px.pie(df, values=df['churn_risk_score'].value_counts(), 
             names=['Churn','Not Churn'], title='Customers')
    st.plotly_chart(fig)


    # Characteristik of membership based on the feedback and churn risk
    st.write('#### Characteristic of Membership Based on The Feedback and Churn Risk')
    fig=plt.figure(figsize=[12,12])
    plt.subplot(311)
    sns.scatterplot(data=df, x='membership_category', y="feedback")

    plt.subplot(312)
    sns.countplot(x = 'membership_category', hue = 'feedback', data = df)

    plt.subplot(313)
    ax=sns.countplot(x = 'membership_category', hue = 'churn_risk_score', data = df)
    ax.legend(labels = ['Not Churn', 'Churn'])

    st.pyplot(fig)

    # Characteristik of membership based on the region category and complaint status
    st.write('#### Average Transaction value Based on Region or Complaint Status')
    pilihan = st.radio('Choose column : ', ('region_category', 'complaint_status'))
    fig = plt.figure(figsize=(12,5))
    sns.countplot(x='membership_category', hue=pilihan, data=df)
    st.pyplot(fig)

    # Characteristik of membership based on the average time spent, transaction value, login days, and points
    st.write('#### Characteristic of Membership Based on The Average Time Spent, Transaction value, Login Days, and Points')
    fig=plt.figure(figsize=[12,20])
    plt.subplot(411)
    sns.boxplot(x = 'membership_category', y = 'avg_time_spent', data = df, showfliers = False)
    plt.grid(True)

    plt.subplot(412)
    sns.boxplot(x = 'membership_category', y = 'avg_transaction_value', data = df, showfliers = False)
    plt.grid(True)

    plt.subplot(413)
    sns.boxplot(x = 'membership_category', y = 'avg_frequency_login_days', data = df, showfliers = True)
    plt.grid(True)

    plt.subplot(414)
    sns.boxplot(x = 'membership_category', y = 'points_in_wallet', data = df, showfliers = False)
    plt.grid(True)

    st.pyplot(fig)

    # Categorize membership based on medium of operation
    st.write('#### Categorize Membership Based on Medium of Operation')
    fig=plt.figure(figsize=[12,5])
    sns.countplot(x = 'membership_category', hue = 'medium_of_operation', data = df)
    st.pyplot(fig)


if __name__ == '__main__':
    run()