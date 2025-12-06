from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from tensorflow.keras.models import load_model, Sequential
import numpy as np
from datetime import datetime
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# MongoDB setup
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.server_info()
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit(1)

db = client['heart_disease_db']
users_collection = db['users']
predictions_collection = db['predictions']

# Load models and scaler
try:
    ann_model = load_model('ann_model.h5')
    with open('dt_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Models and scaler loaded successfully")
except Exception as e:
    print(f"Error loading models or scaler: {e}")
    exit(1)

@app.route('/')
def index():
    print("Rendering login.html")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            mobile = request.form['mobile']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirmPassword']

            if password != confirm_password:
                flash('Passwords do not match')
                return render_template('login.html')

            if users_collection.find_one({'email': email}):
                flash('Email already registered')
                return render_template('login.html')

            result = users_collection.insert_one({
                'name': name,
                'mobile': mobile,
                'email': email,
                'password': password
            })
            print(f"User registered: {email}, ID: {result.inserted_id}, Password: {password}")
            flash('Registration successful! Please log in.')
            return redirect(url_for('index'))
        except Exception as e:
            print(f"Error during registration: {e}")
            flash(f"Registration failed: {str(e)}")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']
            print(f"Login attempt: Email: {email}, Password: {password}")
            user = users_collection.find_one({'email': email})

            if user:
                stored_password = user['password']
                print(f"User found: {email}, Stored Password: {stored_password}")
                if password == stored_password:
                    print(f"Password match for {email}! Setting session and redirecting to input_form")
                    session['email'] = email
                    return redirect(url_for('input_form'))
                else:
                    print(f"Password does not match for {email}. Expected: {stored_password}, Got: {password}")
                    flash('Invalid credentials')
            else:
                print(f"No user found with email: {email}")
                flash('Invalid credentials')
            return render_template('login.html')
        except Exception as e:
            print(f"Error during login: {e}")
            flash(f"Login failed: {str(e)}")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/input_form')
def input_form():
    if 'email' not in session:
        print("No session email, redirecting to login")
        flash('Please log in first')
        return redirect(url_for('index'))
    print(f"Session email: {session['email']}. Rendering input_form.html")
    return render_template('input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        flash('Please log in first')
        return redirect(url_for('index'))

    try:
        features = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            int(request.form['trestbps']),
            int(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            int(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]
        features_scaled = scaler.transform([features])
        feature_extractor = Sequential(ann_model.layers[:-1])
        extracted_features = feature_extractor.predict(features_scaled)
        prediction = dt_model.predict(extracted_features)[0]
        result = "Heart Failure" if prediction == 1 else "No Heart Failure"

        predictions_collection.insert_one({
            'email': session.get('email', 'unknown'),
            'features': features,
            'prediction': result,
            'timestamp': datetime.now()
        })
        print(f"Prediction stored for {session['email']}: {result}")
        return render_template('input_form.html', prediction=result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        flash(f"Prediction failed: {str(e)}")
        return render_template('input_form.html')

if __name__ == '__main__':
    app.run(debug=True)