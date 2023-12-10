from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from preprocessing import preprocess_text
import os

app = Flask(__name__)


# Specify the file paths
model_filename = os.path.join('models', 'ramdomized_logreg_model.pkl')
tfidf_vectorizer_filename = os.path.join('models', 'tfidf_vectorizer.pkl')
xtrain_tfidf_filename = os.path.join('models', 'xtrain_tfidf.pkl')
xvalid_tfidf_filename = os.path.join('models', 'xvalid_tfidf.pkl')

# Load the pre-trained logistic regression model
with open(model_filename, 'rb') as file:
    best_logreg_model = pickle.load(file)

# Load the pre-trained TF-IDF vectorizer
with open(tfidf_vectorizer_filename, 'rb') as file:
    tfidf_vect = pickle.load(file)

# Load the transformed matrices
with open(xtrain_tfidf_filename, 'rb') as file:
    xtrain_tfidf = pickle.load(file)

with open(xvalid_tfidf_filename, 'rb') as file:
    xvalid_tfidf = pickle.load(file)

# Define label_mapping
label_mapping = {
    "Credit reporting, credit repair services, or other personal consumer reports": 0,
    "Credit card or prepaid card": 1,
    "Mortgage": 2,
    "Debt collection": 3,
    "Checking or savings account": 4,
    "Payday loan, title loan, or personal loan": 5,
    "Vehicle loan or lease": 6,
    "Money transfer, virtual currency, or money service": 7,
    "Student loan": 8,
    "Consumer Loan": 9,
    "Bank account or service": 10
}

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict page
@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
        consumer_complaint_narrative = request.form['complaint']
        preprocessed_text = preprocess_text(consumer_complaint_narrative)
        x_input_tfidf = tfidf_vect.transform([preprocessed_text])  # Use the pre-fitted vectorizer
        prediction = best_logreg_model.predict(x_input_tfidf)
        result = list(label_mapping.keys())[list(label_mapping.values()).index(prediction[0])]
        return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(port=8080)
