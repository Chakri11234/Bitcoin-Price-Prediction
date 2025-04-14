from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model and the scaler
model = load_model("C:/Users/chakr/Desktop/Bitcoin_price_prediction/Bitcoin-Price-Prediction/bitcoin_lstm_model.h5")
scaler = joblib.load("C:/Users/chakr/Desktop/Bitcoin_price_prediction/Bitcoin-Price-Prediction/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the 'Close' price from the form
        close_price = float(request.form['close'])

        # Scale the 'Close' price
        scaled_close = scaler.transform(np.array([[close_price]]))

        # Reshape the scaled input for LSTM model
        scaled_close = scaled_close.reshape(1, 1, 1)

        # Predict the next 'Close' price
        prediction = model.predict(scaled_close)

        # Inverse transform the predicted value to get the actual predicted price
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # Make a decision (Buy or Sell)
        if predicted_price > close_price:
            decision = "Buy"
        else:
            decision = "Sell"

        return render_template('index.html', prediction=decision, predicted_price=round(predicted_price, 2))

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
