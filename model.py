import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
crime_data = pd.read_csv("crime_data.csv")

# Convert Time into Hour
crime_data['Hour'] = pd.to_datetime(
    crime_data['Time']
).dt.hour

# Create Risk Score
crime_data['Risk_Score'] = (
    crime_data['Crime_Count'] * 2 +
    crime_data['Night_Crime'] * 3 +
    crime_data['Violent_Crime'] * 5
)

# Encode Text Columns
area_encoder = LabelEncoder()
crime_encoder = LabelEncoder()
level_encoder = LabelEncoder()

crime_data['Area'] = area_encoder.fit_transform(
    crime_data['Area']
)

crime_data['Crime_Type'] = crime_encoder.fit_transform(
    crime_data['Crime_Type']
)

crime_data['Crime_Level'] = level_encoder.fit_transform(
    crime_data['Crime_Level']
)

# Features
X = crime_data[[
    'Area',
    'Crime_Type',
    'Latitude',
    'Longitude',
    'Crime_Count',
    'Night_Crime',
    'Violent_Crime',
    'Hour',
    'Risk_Score'
]]

# Target
y = crime_data['Crime_Level']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100
)

# Train Model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model Accuracy:", accuracy)

# Save Model
joblib.dump(
    model,
    'crime_model.pkl'
)

print("Model Saved Successfully")