import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Use the new caching method (st.cache_data) to load the data
@st.cache_data
def load_data():
    data_url = 'https://raw.githubusercontent.com/your-username/your-repository-name/main/Report202503141112%20-%20Sheet1.csv'
    data = pd.read_csv(data_url, header=None, names=['Store', 'Product', 'Date', 'Quantity'])
    return data

# Load the data
data = load_data()

# Extract store location from the Store column - FIXED APPROACH
def extract_location(store_name):
    if 'Lidl Ireland Gmbh - ' in store_name:
        return store_name.split('Lidl Ireland Gmbh - ')[1]
    elif 'Lidl Northern Ireland Limited' in store_name:
        return 'Northern Ireland Limited'
    return store_name

data['Location'] = data['Store'].apply(extract_location)

# Convert date to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['DayOfWeek'] = data['Date'].dt.dayofweek  # Monday=0, Sunday=6
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Week'] = data['Date'].dt.isocalendar().week

# Extract product details
data['ProductType'] = 'Baby Corn'  # All products are Baby Corn in this dataset
data['PackageSize'] = data['Product'].str.extract(r'(\d+ x \d+g)')[0]  # Extract first match

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Location', 'DayOfWeek', 'PackageSize'])

# Prepare features
features = ['Day', 'Month', 'Year', 'DayOfYear', 'Week'] + \
           [col for col in data.columns if col.startswith('Location_') or 
            col.startswith('DayOfWeek_') or 
            col.startswith('PackageSize_')]

X = data[features]
y = data['Quantity']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and calculate MAE for RandomForestRegressor
rf_y_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_y_pred)

# Predict and calculate MAE for LinearRegression
lr_y_pred = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_y_pred)

# Compare the models and select the best one
if rf_mae < lr_mae:
    best_model = rf_model
    best_model_name = "Random Forest Regressor"
    best_model_mae = rf_mae
else:
    best_model = lr_model
    best_model_name = "Linear Regression"
    best_model_mae = lr_mae

# Save the best model using joblib
joblib.dump(best_model, 'best_model.pkl')

# Output the best model information
print(f"The Best Model: {best_model_name}")
print(f"Mean Absolute Error for {best_model_name}: {best_model_mae:.2f}")

# Prediction function
def round_to_nearest_10(value):
    return round(value / 10) * 10

def predict_demand(start_date, end_date, location, model):
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
        
        input_df = pd.DataFrame([features])
        
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[X.columns]
        
        pred = model.predict(input_df)
        rounded_pred = round_to_nearest_10(pred[0])
        predictions.append((date, rounded_pred))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Quantity'])

# Function to take user input for date and location
def get_input_and_predict():
    start_date_input = input("Enter the start date (YYYY-MM-DD): ")
    end_date_input = input("Enter the end date (YYYY-MM-DD): ")

    try:
        start_date = datetime.strptime(start_date_input, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_input, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    location = input("Enter store location (Charleville, Mullingar, Newbridge, Northern Ireland Limited): ")

    if location not in ['Charleville', 'Mullingar', 'Newbridge', 'Northern Ireland Limited']:
        print("Invalid location. Please choose from the available locations.")
        return

    print(f"Using {best_model_name} for prediction...")
    print(f"Best Model MAE: {best_model_mae:.2f}")
    print(f"Predicting demand for {location} from {start_date} to {end_date}...")

    predictions = predict_demand(start_date, end_date, location, best_model)
    
    print(f"Predicted demand for {location} from {start_date} to {end_date}:")
    print(predictions.to_string(index=False))

# Call the function to get input and predict
get_input_and_predict()

# To load the model later
# loaded_model = joblib.load('best_model.pkl')
