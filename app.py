from flask import Flask, render_template, request
import pickle
import numpy as np



app = Flask(__name__)

# Load the pre-trained model
with open("model/predictor.pickle","rb") as file:
    model = pickle.load(file)
with open("model/vectorizer.pickle","rb") as file:
    vectorizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    message = request.form['email_text']
    input_data_features = vectorizer.transform([message])
    prediction = model.predict(input_data_features)[0]
    if prediction == 1:
        result =  "✅ Ham Mail (Not Spam)"
        css_class = "success"
    else:
        result = "⚠️ Spam Mail"
        css_class = "danger"

    return render_template('results.html', prediction=result,email_text = message, css_class=css_class)

        
if __name__ == '__main__':
    app.run(debug=True)