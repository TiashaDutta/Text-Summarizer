import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download required NLTK data
nltk.download('punkt')

# Create Flask app
app = Flask(__name__)

# Function to summarize text using LSA
def summarize_text(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        input_text = request.form['input_text']
        summary = summarize_text(input_text)
    return render_template('index.html', summary=summary)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
