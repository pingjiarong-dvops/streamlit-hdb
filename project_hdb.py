import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('trained_lgb_reg_model.pkl')

st.markdown("""
    <div style="text-align: center;">
        <img src="https://cdn.dribbble.com/users/391450/screenshots/3536615/hdb.gif" alt="loading GIF" width="400"/>
    </div>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div style="text-align: center;">
        <h2>🏢 THE HDB RESALE PRICE PREDICTOR 🏢</h2>
        <p>This app predicts the <strong>RESALE PRICE</strong> of a flat!</p>
        <p>Enter the details below for your prediction!</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')

# Function to get user input
def user_input_features():
    town = st.sidebar.selectbox('Town', ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
    
    flat_type = st.sidebar.selectbox('Flat Type', ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
    storey_range = st.sidebar.selectbox('Storey Range', ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'])
    flat_model = st.sidebar.selectbox('Flat Model', ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2'])
    
    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', 30, 200, 80)
    remaining_lease_year = st.sidebar.slider('Remaining Lease (years)', 30, 60, 99)
    
    # Only the relevant user input values
    data = {
        'floor_area_sqm': floor_area_sqm,
        'remaining_lease_year': remaining_lease_year,
        'town': town,
        'flat_type': flat_type,
        'storey_range': storey_range,
        'flat_model': flat_model
    }

    return data

# Get user input
user_data = user_input_features()

# Show user input (display only relevant features)
st.markdown("""
    <div style="background-color: #141414; border-radius: 15px; padding: 20px; margin-top: 20px; color: white; margin-bottom: 10px;">
        <h3>User Input Parameters</h3>
        <p><strong>Town:</strong> {}</p>
        <p><strong>Flat Type:</strong> {}</p>
        <p><strong>Storey Range:</strong> {}</p>
        <p><strong>Flat Model:</strong> {}</p>
        <p><strong>Floor Area (sqm):</strong> {}</p>
        <p><strong>Remaining Lease (years):</strong> {}</p>
    </div>
""".format(user_data['town'], user_data['flat_type'], user_data['storey_range'], user_data['flat_model'], user_data['floor_area_sqm'], user_data['remaining_lease_year']), unsafe_allow_html=True)

# Function to apply one-hot encoding to the user input for model compatibility
def preprocess_input(user_data):
    data = {
        'floor_area_sqm': user_data['floor_area_sqm'],
        'remaining_lease_year': user_data['remaining_lease_year'],
    }
    
    towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    for town_name in towns:
        data[f'town_{town_name}'] = 1 if user_data['town'] == town_name else 0
    
    flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    for flat in flat_types:
        data[f'flat_type_{flat}'] = 1 if user_data['flat_type'] == flat else 0
    
    storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
    for storey in storey_ranges:
        data[f'storey_range_{storey}'] = 1 if user_data['storey_range'] == storey else 0
    
    flat_models = ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']
    for model in flat_models:
        data[f'flat_model_{model}'] = 1 if user_data['flat_model'] == model else 0
    
    return pd.DataFrame(data, index=[0])

# Preprocess the input data
processed_data = preprocess_input(user_data)

# Button to trigger prediction
if st.button('Predict'):
    # Prediction
    prediction = model.predict(processed_data)
    
    # Show the prediction result
    st.subheader('Predicted Resale Price')
    st.write(f"S$ {prediction[0]:,.2f}")
