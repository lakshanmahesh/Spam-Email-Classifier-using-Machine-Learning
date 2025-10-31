# Spam-Email-Classifier-using-Machine-Learning
This project is a Spam Email Detection System built using Python and Scikit-learn.   It classifies incoming emails as either Spam or Ham (Non-Spam) using a TF-IDF vectorizer and a Logistic Regression model.   The model was trained on a labeled SMS/email dataset and deployed through a simple web interface for real-time predictions.

### data set link
https://www.kaggle.com/datasets/abdmental01/email-spam-dedection

## 🚀 Features
- Preprocessed and vectorized text using TF-IDF
- Trained and tuned Logistic Regression model
- Achieved 98.42% accuracy after hyperparameter tuning
- Saved model and vectorizer using Pickle
- Built a web interface for end-user prediction

## 🧠 Model Training
    from sklearn.feature_extraction.text import TfidfVectorizer
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_features, y_train)
## 💾 Model Saving
    import pickle
    with open('model.pickle', 'wb') as file:
      pickle.dump(model, file)
      
    with open('vectorizer.pickle', 'wb') as file:
      pickle.dump(feature_extraction, file)

## 🌐 Web Interface

A lightweight web app (built with Flask) allows users to:

Enter an email or message.

Click Predict to check if it’s spam or ham.

## 🧰 Technologies Used

Python

Scikit-learn

Pandas, NumPy

Flask 

Pickle

## ⚙️ Installation
    git clone https:https://github.com/lakshanmahesh/Spam-Email-Classifier-using-Machine-Learning.git
    cd spam-email-classifier
    pip install -r requirements.tx

Then run

    python app.py











