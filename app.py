from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
# Use a more secure secret key in production
app.secret_key = os.urandom(24)

# Ensure the model files exist before loading
MODEL_PATH = 'model1.pkl'
ENCODER_PATH = 'target_encoder.pkl'

# Load the models with error handling
try:
    # Load the trained model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    else:
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

    # Load the target encoder
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, 'rb') as file:
            target_encoder = pickle.load(file)
    else:
        raise FileNotFoundError(f"Encoder file {ENCODER_PATH} not found")

except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Valid cities list for validation
VALID_CITIES = [
    "City_chattogram",
    "City_cumilla",
    "City_dhaka",
    "City_gazipur",
    "City_narayanganj-city"
]

@app.route('/')
@app.route('/home')
def home():
    """Route for the home page"""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Extract form data
        data = {
            "city": request.form.get("City"),
            "location": request.form.get("Location"),
            "bedrooms": request.form.get("Bedrooms"),
            "bathrooms": request.form.get("Bathrooms"),
            "floor_area": request.form.get("Floor_area"),
            "floor_no": request.form.get("Floor_no")
        }

        # Validate all fields are present
        if not all(data.values()):
            return render_template('home.html', 
                                prediction_text="Please fill out all fields.")

        # Validate city
        if data['city'] not in VALID_CITIES:
            return render_template('home.html', 
                                prediction_text="Invalid city selected.")

        # Convert numeric fields with validation
        try:
            numeric_data = {
                "bedrooms": int(data['bedrooms']),
                "bathrooms": int(data['bathrooms']),
                "floor_area": float(data['floor_area']),
                "floor_no": int(data['floor_no'])
            }

            # Additional numeric validation
            if any(value <= 0 for value in numeric_data.values()):
                return render_template('home.html', 
                                    prediction_text="All numeric values must be greater than 0.")

        except ValueError:
            return render_template('home.html', 
                                prediction_text="Invalid numeric input. Please check your numbers.")

        # Create city features dictionary
        city_features = {city: 1 if city == data['city'] else 0 for city in VALID_CITIES}

        # Prepare input DataFrame
        input_data = {
            **city_features,
            "Location": data['location'],
            "Bedrooms": numeric_data['bedrooms'],
            "Bathrooms": numeric_data['bathrooms'],
            "Floor_area": numeric_data['floor_area'],
            "Floor_no": numeric_data['floor_no']
        }

        # Create and arrange DataFrame columns
        input_df = pd.DataFrame([input_data])
        column_order = [
            "Bedrooms", "Bathrooms", "Floor_no", "Floor_area", "Location",
            "City_chattogram", "City_cumilla", "City_dhaka", "City_gazipur", 
            "City_narayanganj-city"
        ]
        input_df = input_df[column_order]

        # Transform location using target encoder
        input_df['Location'] = target_encoder.transform(input_df[['Location']])

        # Make prediction
        prediction = model.predict(input_df)
        predicted_price = f"{prediction[0]:,.2f}"

        # Return prediction template with all data
        return render_template('prediction.html',
                            city=data['city'].replace('City_', '').title(),
                            location=data['location'],
                            bedrooms=numeric_data['bedrooms'],
                            bathrooms=numeric_data['bathrooms'],
                            floor_area=numeric_data['floor_area'],
                            floor_no=numeric_data['floor_no'],
                            predicted_price=predicted_price)

    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error during prediction: {e}")
        return render_template('home.html', 
                            prediction_text="An error occurred during prediction. Please try again.")

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('home.html', 
                         prediction_text="Page not found. Please try again."), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('home.html', 
                         prediction_text="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True)