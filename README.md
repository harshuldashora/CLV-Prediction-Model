# Customer Lifetime Value (CLV) Prediction  

## Overview  
This project predicts **Customer Lifetime Value (CLV)** using Amazon customer purchase data. By analyzing purchase behavior, discounts, payment methods, and customer segments, the model helps businesses understand long-term customer value and make better decisions for retention and marketing.  

## Workflow  
- The raw data is cleaned by fixing missing values, removing duplicates, and handling outliers.  
- New features are created, including loyalty score, customer segment (New, Regular, VIP), discount usage, return status, and shopping channel preference.  
- A machine learning pipeline is built with preprocessing for numeric and categorical features.  
- A linear regression model is trained to estimate customer lifetime value.  
- The model is evaluated using R² and RMSE scores and then saved for future use.  

## Files  
- **cleaned_amazon_data.csv** – Preprocessed dataset.  
- **clv_model_pipeline.pkl** – Trained prediction model.  
- **model_features.pkl** – Features used for prediction.  
- **main.py** – Script for data cleaning, feature engineering, and model training.  
- **app.py** – Flask app to run predictions through a web interface.  
- **templates/index.html** – Web form for entering customer details.  
- **requirements.txt** – Project dependencies.  

## Running the Project  

### Train the Model  
```bash
python main.py
