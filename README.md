import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset (You can use a real credit card dataset like the one from Kaggle)
# For demo purposes, let's assume 'creditcard.csv' is available
data = pd.read_csv("creditcard.csv")

# Step 1: Preprocess Data
X = data.drop(['Class'], axis=1)  # Features
y = data['Class']  # Target: 0 = Normal, 1 = Fraud

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train AI Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict
predictions = model.predict(X_test)

# Step 5: Evaluate
print("Evaluation Report:")
print(classification_report(y_test, predictions))

# Step 6: Grade New Transactions
def grade_transaction(transaction):
    pred = model.predict([transaction])[0]
    if pred == 1:
        return "Fraudulent"
    else:
        return "Legitimate"

# Example: Predict a new transaction (use real values from your dataset)
# Sample format: [V1, V2, ..., V28, Amount]
sample_transaction = X_test.iloc[0].tolist()
grade = grade_transaction(sample_transaction)
print("Transaction Grade:", grade)
