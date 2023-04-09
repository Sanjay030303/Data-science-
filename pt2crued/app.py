
from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('model (2).h5')

# Define a function to predict the crude oil price for the next day
def predict_price(model, time_step, input_data):
    # Scale the input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(np.array(input_data).reshape(-1, 1))

    # Reshape the input data for the LSTM model
    input_data = input_data.reshape(1, time_step, 1)

    # Make the prediction
    prediction = model.predict(input_data)

    # Inverse scale the prediction
    prediction = scaler.inverse_transform(prediction)

    return prediction[0][0]

# Initialize Flask app
app = Flask(__name__, static_folder='static')


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/home/')
def index():
    return render_template('home.html')
# Define route for prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input for the last 10 days' prices
    last_10_days = request.form['last_10_days']
    last_10_days = [x.strip() for x in last_10_days.split(',')]

    # Check if the input is valid
    if len(last_10_days) != 10:
        return render_template('index.html', error='Please enter exactly 10 prices')

    for price in last_10_days:
        if not price:
            return render_template('index.html', error='Please enter valid prices')

    last_10_days = [float(x) for x in last_10_days]

    # Make a prediction for the next day's price
    next_day_price = predict_price(model, time_step=10, input_data=last_10_days)
    return render_template('index.html', prediction=f'{next_day_price:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
