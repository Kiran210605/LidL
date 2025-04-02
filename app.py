import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/your-username/your-repository-name/main/Report202503141112%20-%20Sheet1.csv', header=None, names=['Store', 'Product', 'Date', 'Quantity'])
    return data

data = load_data()

# Extract store location from the Store column
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

# Round to nearest 10
def round_to_nearest_10(value):
    return round(value / 10) * 10

# Function to predict demand
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
        
        # Set other locations to 0
        for loc in ['Charleville', 'Mullingar', 'Newbridge', 'Northern Ireland Limited']:
            if loc != location:
                features[f'Location_{loc}'] = 0
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Ensure all columns are present
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[X.columns]
        
        # Predict
        pred = model.predict(input_df)
        rounded_pred = round_to_nearest_10(pred[0])  # Round to nearest 10
        predictions.append((date, rounded_pred))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Quantity'])

# Streamlit App Interface
st.title("Product Demand Prediction")
st.write(f"Mean Absolute Error of Random Forest Regressor: {rf_mae:.2f}")
st.write(f"Mean Absolute Error of Linear Regression: {lr_mae:.2f}")

start_date = st.date_input("Select start date", datetime.now())
end_date = st.date_input("Select end date", datetime.now() + timedelta(days=6))

location = st.selectbox("Select Store Location", ['Charleville', 'Mullingar', 'Newbridge', 'Northern Ireland Limited'])

model_choice = st.radio("Choose Prediction Model", ("Random Forest Regressor", "Linear Regression"))

if st.button("Predict"):
    if model_choice == "Random Forest Regressor":
        model = rf_model
    else:
        model = lr_model
    
    predictions = predict_demand(start_date, end_date, location, model)
    st.write(f"Predicted demand for {location} from {start_date} to {end_date}:")
    st.dataframe(predictions)

    # Displaying MAE (Accuracy)
    if model_choice == "Random Forest Regressor":
        st.write(f"Accuracy (Mean Absolute Error) of Random Forest: {rf_mae:.2f}")
    elif model_choice == "Linear Regression":
        st.write(f"Accuracy (Mean Absolute Error) of Linear Regression: {lr_mae:.2f}")
