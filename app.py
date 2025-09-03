from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load("clv_model_pipeline.pkl")
features = joblib.load("model_features.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = {}
    input_data['Age'] = int(data.get('Age', 0))
    input_data['Purchase_Amount'] = float(data.get('Purchase_Amount', 0))
    input_data['Loyalty_Score'] = float(data.get('Loyalty_Score', 0))
    input_data['Discount_Applied'] = data.get('Discount_Applied', 'No')
    input_data['Payment_Method'] = data.get('Payment_Method', 'Other')
    input_data['Customer_Segment'] = data.get('Customer_Segment', 'New')
    input_data['Preferred_Shopping_Channel'] = data.get('Preferred_Shopping_Channel', 'Online')
    df = pd.DataFrame([input_data])
    prediction = pipeline.predict(df)[0]
    return f"Predicted Customer Lifetime Value: ₹{round(prediction, 2)}"

if __name__ == '__main__':
    app.run(debug=True)
