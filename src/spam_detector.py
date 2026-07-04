import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the dataset
mail_data = pd.read_csv("data/spam.csv")

# Display the first five rows
# Display the first five rows
print(mail_data.head())

# Display dataset information
print("\nDataset Information:")
print(mail_data.info())

# Check for missing values
print("\nMissing Values:")
print(mail_data.isnull().sum())

# Display dataset shape
print("\nDataset Shape:")
print(mail_data.shape)
# Convert labels to numerical values
mail_data["Category"] = mail_data["Category"].map({
    "ham": 0,
    "spam": 1
})

# Separate features and labels
X = mail_data["Message"]
y = mail_data["Category"]

print("\nFirst 5 Labels:")
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print("\nText has been converted into numerical features.")

# Train the Logistic Regression model
model = LogisticRegression()

model.fit(X_train_features, y_train)
# Save the trained model
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully!")

print("Model trained successfully!")

# Make predictions on the test data
y_pred = model.predict(X_test_features)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# Predict a new email
# ==========================================

# Get email text from the user
input_mail = input("\nEnter an email message: ")

# Convert the input into numerical features
input_data_features = vectorizer.transform([input_mail])

# Predict
prediction = model.predict(input_data_features)

# Display the result
if prediction[0] == 1:
    print("\nPrediction: Spam Email")
else:
    print("\nPrediction: Ham (Not Spam)")