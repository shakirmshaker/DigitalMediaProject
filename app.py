import streamlit as st
import pandas as pd
import pickle
from faker import Faker
import time

# Function to load the machine learning model from a pickled file
@st.cache_data
def load_model():
    with open('models/mlModel.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to load the customer data
@st.cache_data
def load_data():    
    X_test = pd.read_csv('data/test_data/X_test.csv')
    y_test = pd.read_csv('data/test_data/y_test.csv')
    data = pd.read_csv('data/raw_data/TelcoCustomerChurn.csv')
    return X_test, y_test, data

# Function to predict churn probability for a customer
def predict_churn(model, customer_data):
    probability = model.predict_proba(customer_data)[:, 1]
    return probability

# Main application function
def main():
    st.sidebar.title('Customer Churn Prediction')
    
    # Load model and data
    model = load_model()
    X_test, y_test, data = load_data()
    
    # Check if the state has the customer names. If not, generate them
    if 'customer_names' not in st.session_state:
        fake = Faker()
        st.session_state.customer_names = [fake.name() for _ in range(len(X_test))]
    
    # Select a customer    
    customer_name = st.sidebar.selectbox('Select a customer', st.session_state.customer_names)
    customer_idx = st.session_state.customer_names.index(customer_name)
    customer_data = X_test.iloc[customer_idx].to_frame().T
    customer_id = customer_data['CustomerID'][customer_idx]    
    
    # Ensure the selected data is in the correct format for prediction
    customer_data_for_prediction = X_test.iloc[customer_idx].drop(labels=['CustomerID']).astype(float).values.reshape(1, -1)
    
    # Predict and display churn probability
    probability = predict_churn(model, customer_data_for_prediction)[0]
    probability = round(probability * 100, 2)
    
    # Display customer profile
    st.title(f'{customer_name}')
    st.write('---')

    st.header('Churn Probability')
    st.header(' ')

    with st.spinner('Estimating Churn Probability') as status:
        time.sleep(1.5)
        
        if probability <= 50:
            st.success(f'Estimate: {probability} %')
        if 50 < probability <= 70 :
            st.warning(f'Estimate: {probability} %')
        if probability > 70:
            st.error(f'Estimate: {probability} %')
        st.toast('Successfully predicted customer churn!')
            
        
    # Display Data in a more visual manner
    customer_raw_data = data[data['customerID'] == customer_id].reset_index()

    gender = customer_raw_data['gender'][0]
    Partner = customer_raw_data['Partner'][0]
    Contract = customer_raw_data['Contract'][0]
    DeviceProtection = customer_raw_data['DeviceProtection'][0]
    InternetService = customer_raw_data['InternetService'][0]
    MonthlyCharges = customer_raw_data['MonthlyCharges'][0]
    MultipleLines = customer_raw_data['MultipleLines'][0]
    OnlineBackup = customer_raw_data['OnlineBackup'][0]
    OnlineSecurity = customer_raw_data['OnlineSecurity'][0]
    PaperlessBilling = customer_raw_data['PaperlessBilling'][0]    
    PaymentMethod = customer_raw_data['PaymentMethod'][0]
    PhoneService = customer_raw_data['PhoneService'][0]
    SeniorCitizen = customer_raw_data['SeniorCitizen'][0]
    StreamingMovies = customer_raw_data['StreamingMovies'][0]
    StreamingTV = customer_raw_data['StreamingTV'][0]
    TechSupport = customer_raw_data['TechSupport'][0]
    tenure = customer_raw_data['tenure'][0]

    st.header('Customer information')
    st.header(' ')
    ex = st.expander('Customer information')

    with ex:
        
        col1, col2 = st.columns(2)    

        with col1:
            st.subheader(f'Contract', help = 'The contract term of the customer (Month-to-month, One year, Two year)')
            st.write(Contract)

            st.subheader(f'DeviceProtection',  help = 'Whether the customer has device protection or not (Yes, No, No internet service)')
            st.write(DeviceProtection)

            st.subheader(f'MonthlyCharges',  help = 'The amount charged to the customer monthly')
            st.write(str(MonthlyCharges))

            st.subheader(f'OnlineSecurity',  help = 'Whether the customer has online security or not (Yes, No, No internet service)')
            st.write(OnlineSecurity)

            st.subheader(f'PaymentMethod', help = 'The customer\'s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))')
            st.write(PaymentMethod)

            st.subheader(f'StreamingMovies', help = 'Whether the customer has streaming movies or not (Yes, No, No internet service)')
            st.write(StreamingMovies)

        with col2:
            st.subheader(f'InternetService', help = 'Customer\'s internet service provider (DSL, Fiber optic, No)')
            st.write(InternetService)

            st.subheader(f'MultipleLines', help = 'Whether the customer has multiple lines or not (Yes, No, No phone service)')
            st.write(MultipleLines)

            st.subheader(f'OnlineBackup', help = 'Whether the customer has online backup or not (Yes, No, No internet service)')
            st.write(OnlineBackup)

            st.subheader(f'PaperlessBilling', help = 'Whether the customer has paperless billing or not (Yes, No)')
            st.write(PaperlessBilling)
            
            st.subheader(f'PhoneService', help = 'Whether the customer has a phone service or not (Yes, No)')
            st.write(PhoneService)
            
            st.subheader(f'TechSupport', help = 'Whether the customer has tech support or not (Yes, No, No internet service)')
            st.write(TechSupport)

    st.header('Model information')

    feature_importance_df = pd.read_csv('data/preprocessed_data/feature_importance.csv')
    feature_importance_df = feature_importance_df.head(10)
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True).set_index('Feature')
    
    # Plot

    st.write('In churn prediction, the absolute values of feature importance indicate the strength of each feature\'s impact on churn likelihood. Higher absolute values mean greater influence. Below are top 10 influential features shown.')
    st.header('')
    st.bar_chart(feature_importance_df)

    
# Run the app
if __name__ == "__main__":
    main()
