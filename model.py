import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv("data.csv")

# Convert text → numbers
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Smoking'] = data['Smoking'].map({'Yes': 1, 'No': 0})
data['Diabetes'] = data['Diabetes'].map({'Yes': 1, 'No': 0})
data['Hypertension'] = data['Hypertension'].map({'Yes': 1, 'No': 0})
data['Immediate_Loading'] = data['Immediate_Loading'].map({'Yes': 1, 'No': 0})
data['Jaw_Location'] = data['Jaw_Location'].map({'Upper': 1, 'Lower': 0})
data['Bone_Density'] = data['Bone_Density'].map({'D1':1,'D2':2,'D3':3,'D4':4})

# Remove missing values
data = data.dropna()

# Split data
X = data.drop("Implant_Outcome", axis=1)
y = data["Implant_Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=5)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cross validation
scores = cross_val_score(model, X, y, cv=5)
print("CV Score:", scores.mean())

# Save model + feature names
pickle.dump({"model": model, "features": list(X.columns)}, open("model.pkl", "wb"))

print("MODEL READY")