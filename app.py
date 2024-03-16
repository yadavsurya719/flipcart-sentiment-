from flask import Flask, render_template, request
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import unicodedata
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

model_path = 'C:\\Users\\achin\\Desktop\\New folder\\model\\Innomatics_sentiment_model.joblib'


def clean_text(sentence):
    if not isinstance(sentence, str):
        return ""
    text = sentence.lower()

    text = BeautifulSoup(text, 'html.parser').get_text()

    text = ' '.join([word for word in text.split() if not word.startswith('http')])

    text = ''.join([char for char in text if char not in string.punctuation + '’‘'])

    text = ''.join([i for i in text if not i.isdigit()])

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    return text


model = joblib.load(model_path)


def predict_sentiment(text):
    text = clean_text(text)
    prediction = model.predict([text])
    return prediction


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['text']
        prediction = predict_sentiment(user_input)
        return render_template('result.html', user_input=user_input, prediction=prediction)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
