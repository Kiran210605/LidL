import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

# Load the saved model
best_model = joblib.load('best_model.pkl')

# Load the saved feature columns (this was saved during training)
with open('X_columns.pkl', 'rb') as f:
    X_columns = joblib.load(f)

# Function to round predicted values to the nearest 10
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

        # Set other locations to 0
        for loc in ['Charleville', 'Mullingar', 'Newbridge', 'Northern Ireland Limited']:
            if loc != location:
                features[f'Location_{loc}'] = 0

        # Create a DataFrame from the features
        input_df = pd.DataFrame([features])

        # Ensure all the expected columns (from training) are present in the input_df
        missing_columns = [col for col in X_columns if col not in input_df.columns]
        for col in missing_columns:
            input_df[col] = 0  # Add missing columns with value 0

        # Reorder the input_df columns to match the training data column order
        input_df = input_df[X_columns]

        # Debugging step: print columns and their order for verification
        print("Input columns for prediction:", input_df.columns.tolist())
        print("Expected training columns:", X_columns)

        # Make the prediction with the model
        pred = model.predict(input_df)

        # Round the prediction to the nearest 10
        rounded_pred = round_to_nearest_10(pred[0])
        predictions.append((date, rounded_pred))

    return pd.DataFrame(predictions, columns=['Date', 'Predicted Quantity'])

# Streamlit interface
st.title("Product Demand Prediction")

# Input form for the user
start_date_input = st.date_input("Enter the start date")
end_date_input = st.date_input("Enter the end date")
location = st.selectbox("Select store location", ["Charleville", "Mullingar", "Newbridge", "Northern Ireland Limited"])

if st.button("Predict Demand"):
    if start_date_input and end_date_input and location:
        # Ensure the dates are in the correct format
        start_date = start_date_input
        end_date = end_date_input

        st.write(f"Predicting demand for {location} from {start_date} to {end_date}...")

        # Get predictions
        predictions = predict_demand(start_date, end_date, location, best_model, X_columns)

        # Show the predicted demand
        st.write("Predicted demand:")
        st.dataframe(predictions)
    else:
        st.error("Please provide all inputs.")
