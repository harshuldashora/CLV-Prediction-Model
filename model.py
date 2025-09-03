import pandas as pd
import numpy as np
from scipy import stats
import joblib

df = pd.read_csv(r"C:\Users\HARSHUL DASHORA\OneDrive\Documents\Copy of Amazon_Customer_Purchase_Data (2)(1).csv")
for col in ['Age', 'Purchase_Amount', 'Rating', 'Customer_Lifetime_Value']:
    median = df[col].median()
    df[col].fillna(median, inplace=True)
mode_payment = df['Payment_Method'].mode()[0]
df['Payment_Method'].fillna(mode_payment, inplace=True)
df.drop_duplicates(subset=['Customer_ID', 'Purchase_Date'], inplace=True)
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
df['Gender'] = (df['Gender'].str.strip().str.capitalize()
                   .replace({'M': 'Male', 'F': 'Female', 'Other': 'Other'}))
df['Age'] = df['Age'].astype(int)
df['Purchase_Amount'] = df['Purchase_Amount'].astype(float)

for col in ['Purchase_Amount', 'Customer_Lifetime_Value']:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)
    z_scores = np.abs(stats.zscore(df[['Purchase_Amount', 'Customer_Lifetime_Value']]))
mask = (z_scores < 3).all(axis=1)
df = df.loc[mask]
df['Customer_Lifetime_Value'] = df.groupby('Customer_ID')['Purchase_Amount'].transform('sum')
frequency = df.groupby('Customer_ID')['Purchase_Date'].transform('count')
total_spent = df['Customer_Lifetime_Value']
df['Loyalty_Score'] = (frequency.rank(pct=True) + total_spent.rank(pct=True)) / 2
df['Discount_Applied'] = np.where(df.get('Discount', 0) > 0, 'Yes', 'No')
if 'Returned' in df.columns:
    df['Return_Status'] = df['Returned'].map({1: 'Yes', 0: 'No'})
else:
    df['Return_Status'] = 'No'
bins = [0, 0.33, 0.66, 1.0]
labels = ['New', 'Regular', 'VIP']
df['Customer_Segment'] = pd.cut(df['Loyalty_Score'], bins=bins, labels=labels)
df['Preferred_Shopping_Channel'] = df.get('Channel', 'Online')
df.to_csv('cleaned_amazon_data.csv', index=False)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

features = [
    'Age', 'Purchase_Amount', 'Loyalty_Score',
    'Discount_Applied', 'Payment_Method',
    'Customer_Segment', 'Preferred_Shopping_Channel'
]
target = 'Customer_Lifetime_Value'

X = df[features]
y = df[target]

numeric_features = ['Age', 'Purchase_Amount', 'Loyalty_Score']
categorical_features = [
    'Discount_Applied', 'Payment_Method', 'Customer_Segment', 'Preferred_Shopping_Channel'
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R² Score:", r2)
print("RMSE:", rmse)
joblib.dump(pipeline, 'clv_model_pipeline.pkl')
joblib.dump(features, 'model_features.pkl')
