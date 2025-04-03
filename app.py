def predict_demand(start_date, end_date, location, model, X_columns):
    date_range = pd.date_range(start=start_date, end=end_date)
    predictions = []
    
    for date in date_range:
        # Feature generation
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
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder the input_df columns to match the training data column order
        input_df = input_df[X_columns]
        
        # Make the prediction with the model
        pred = model.predict(input_df)
        
        # Round the prediction to the nearest 10
        rounded_pred = round_to_nearest_10(pred[0])
        predictions.append((date, rounded_pred))
    
    return pd.DataFrame(predictions, columns=['Date', 'Predicted Quantity'])
