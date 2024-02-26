import numpy 
import nltk
from flask import Flask, render_template, request
import pickle
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

model_file_path = "models/Logistic_Regression_model.pkl"
with open(model_file_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
    

nltk.download('wordnet')   
nltk.download('stopwords')
stopword = set(stopwords.words("english"))

def preprocess_text(text):
    remove_punc = (char if char not in string.punctuation else ' ' for char in text)
    clean_words = "".join(remove_punc)
    text = [word.lower() for word in clean_words.split() if word.lower() not in stopword]
    return text

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    lemmatized_text = " ".join([lemmatizer.lemmatize(word) for word in text])
    return lemmatized_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Preprocess and vectorize the input text
        preprocessed_text = preprocess_text(news_text)
        lemmatized_text = lemmatize_text(preprocessed_text)
        text_vector = vectorizer.transform([lemmatized_text])
        
        # Make prediction using the loaded model
        prediction = loaded_model.predict(text_vector)
        
        model_name = "Logistic Regression"

        if prediction[0] == 1:
            result = "Real News"
        else:
            result = "Fake News"

        print(f"Model: {model_name}")
        print("Prediction:", result)

        return render_template('index.html', prediction=result, news_text=news_text)
    else:
        return "Method not allowed", 405

if __name__ == '__main__':
    app.run(debug=True)
