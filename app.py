import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def train_and_save_model():
    print(f"Current working directory: {os.getcwd()}")
    data = pd.read_csv('salary_data.csv')
    X = data['YearsExperience'].values.reshape(-1, 1)
    y = data['Salary'].values

    # Use polynomial features to capture non-linear relationships
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    with open('salary_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved as 'salary_model.pkl'")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual')
    
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', label='Predicted')
    
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Salary vs Years of Experience')
    plt.legend()
    plt.savefig('static/salary_prediction_plot.png')
    print("Plot saved as 'static/salary_prediction_plot.png'")

# Load or train the model
if not os.path.exists('salary_model.pkl'):
    train_and_save_model()

# Load the model
with open('salary_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    years_experience = float(request.form['years'])
    prediction = model.predict([[years_experience]])[0]
    return jsonify({'prediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
