import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the best model
best_model = joblib.load('best_model.pkl')

# Define the feature extraction function
def extract_location(store_name):
    if 'Lidl Ireland Gmbh - ' in store_name:
        return store_name.split('Lidl Ireland Gmbh - ')[1]
    elif 'Lidl Northern Ireland Limited' in store_name:
        return 'Northern Ireland Limited'
    return store_name

def round_to_nearest_10(value):
    return round(value / 10) * 10

def predict_demand(start_date, end_date, location, model, X_columns):
    date_range = pd.date_range(start=start_date, end=end_date)
    predictions = []
    
    for date in date_range:
        features = {
            'Day': date.day,
            'Month': date.month,
            'Year': date.year,
            'DayOfYear': date.dayofyear,
            'Week': date.isocalendar().week,
            'DayOfWeek_0': 1 if date.dayofweek == 0 else 0,
            'DayOfWeek_1': 1 if date.dayofweek == 1 else 0,
            'DayOfWeek_2': 1 if date.dayofweek == 2 else 0,
            'DayOfWeek_3': 1 if date.dayofweek == 3 else 0,
            'DayOfWeek_4': 1 if date.dayofweek == 4 else 0,
            'DayOfWeek_5': 1 if date.dayofweek == 5 else 0,
            'DayOfWeek_6': 1 if date.dayofweek == 6 else 0,
            'PackageSize_14 x 170g': 0,  # Default to 16x130g
            'PackageSize_16 x 130g': 1,
            f'Location_{location}': 1
        }
        
        for loc in ['Charleville', 'Mullingar', 'Newbridge', 'Northern Ireland Limited']:
            if loc != location:
                features[f'Location_{loc}'] = 0
        
        # Create DataFrame for input data
        input_df = pd.DataFrame([features])
        
        # Ensure all expected columns are present in input_df
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[X_columns]
        
        # Predict with the trained model
        pred = model.predict(input_df)
        rounded_pred = round_to_nearest_10(pred[0])  # Round to nearest 10
        predictions.append((date, rounded_pred))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Quantity'])


# Streamlit user interface
st.title("Product Demand Prediction")

start_date_input = st.date_input("Enter the start date")
end_date_input = st.date_input("Enter the end date")
location = st.selectbox("Select store location", ["Charleville", "Mullingar", "Newbridge", "Northern Ireland Limited"])

if st.button("Predict Demand"):
    if start_date_input and end_date_input and location:
        # Ensure the dates are in the correct format
        start_date = start_date_input
        end_date = end_date_input

        st.write(f"Predicting demand for {location} from {start_date} to {end_date}...")

        # Load the saved model and extract feature names (you should load X.columns if you saved them)
        X_columns = ['Day', 'Month', 'Year', 'DayOfYear', 'Week', 'DayOfWeek_0', 'DayOfWeek_1', 'DayOfWeek_2', 
                     'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'PackageSize_14 x 170g', 
                     'PackageSize_16 x 130g', 'Location_Charleville', 'Location_Mullingar', 'Location_Newbridge', 
                     'Location_Northern Ireland Limited']
        
        # Get predictions
        predictions = predict_demand(start_date, end_date, location, best_model, X_columns)
        
        st.write("Predicted demand:")
        st.dataframe(predictions)
    else:
        st.error("Please provide all inputs.")
