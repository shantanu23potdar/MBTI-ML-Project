import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib  # Import joblib for saving the model
from sklearn import metrics

# Load the dataset
data_set = pd.read_csv("C:/MBTI_106K.csv")

# Assuming 'posts' is the column containing text data and 'type' is the target column
X = data_set['posts']
y = data_set['type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Create a simple pipeline with TfidfVectorizer and Logistic Regression
model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=1000))])  # Increase max_iter

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')

# Save the trained model to a file using joblib
joblib.dump(model, 'personality_model.pkl', compress=1)


