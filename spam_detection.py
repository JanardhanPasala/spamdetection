from flask import Flask, render_template, url_for, request
import joblib
import sys

sys.dont_write_bytecode = True
# PYTHONDONTWRITEBYTECODE=1

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    sms_spam_model = joblib.load('NB_spam_detection.pkl')
    message_vectorizer = joblib.load('count_vectorizer.pkl')
    if request.method=='POST':
        message = request.form['message']
        data = [message]
        message_vector = message_vectorizer.transform(data)
        prediction = sms_spam_model.predict(message_vector)
    return render_template('result.html', spam_prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
