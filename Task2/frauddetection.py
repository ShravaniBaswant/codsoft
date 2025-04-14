import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("C:/Users/Prafull/OneDrive/Desktop/codsoft/fraudTest.csv/fraudTest.csv")

# Drop unnecessary columns
columns_to_drop = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", 
                   "city", "state", "zip", "job", "dob", "trans_num"]
df_cleaned = df.drop(columns=columns_to_drop)

# Encode categorical variables
categorical_cols = ["merchant", "category", "gender"]
for col in categorical_cols:
    df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col])

# Sample dataset to reduce memory usage
df_sampled = df_cleaned.sample(n=100000, random_state=42)

# Split into features and labels
X = df_sampled.drop(columns=["is_fraud"])
y = df_sampled["is_fraud"]

# Normalize numerical features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
