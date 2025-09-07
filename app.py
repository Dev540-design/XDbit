import os
import time
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, g
import requests
import ratelimit
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['DATABASE'] = os.path.join(app.root_path, "chatbot.db")
app.config['SECRET_KEY'] = "dev"

# Download NLTK data if not present
try: nltk.data.find('corpora/wordnet')
except: nltk.download('wordnet')
try: nltk.data.find('corpora/stopwords')
except: nltk.download('stopwords')
try: nltk.data.find('tokenizers/punkt')
except: nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
last_request_time = [time.time()]

ADVANCED_KNOWLEDGE = [
    {"question": "hello", "answer": "Hello there! How can I help you today?"},
    {"question": "how are you", "answer": "I'm just a program, but I'm doing great! Thanks for asking."},
    {"question": "what is your name", "answer": "I am an NLTK Flask Chatbot."},
    {"question": "tell me a joke", "answer": "Why don't scientists trust atoms? Because they make up everything!"},
    {"question": "what time is it", "answer": lambda: f"The current time is {time.strftime('%H:%M:%S')}.", "dynamic": True},
    {"question": "what is the date", "answer": lambda: f"Today's date is {time.strftime('%Y-%m-%d')}.", "dynamic": True},
    {"question": "who is albert einstein", "answer": "Albert Einstein was a theoretical physicist who developed the theory of relativity."},
    {"question": "define artificial intelligence", "answer": "Artificial Intelligence is the simulation of human intelligence in machines."},
    # Add more advanced Q&A pairs here
]

vectorizer = TfidfVectorizer()
knowledge_questions = [item["question"] for item in ADVANCED_KNOWLEDGE]
knowledge_vectors = vectorizer.fit_transform(knowledge_questions)

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            user_input TEXT,
            bot_response TEXT
        );
    """)
    db.commit()

@ratelimit.limits(calls=10, period=60)
def rate_limited_get_url_content(url):
    try:
        current_time = time.time()
        if current_time - last_request_time[0] < 1:
            time.sleep(1 - (current_time - last_request_time[0]))
        last_request_time[0] = time.time()

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        if 'Disallow' in response.text:
            return "Scraping not permitted by website."
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        return f"Error accessing URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred during scraping: {e}"

def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(lemmas)

def custom_semantic_match(user_input):
    processed_input = preprocess(user_input)
    input_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vec, knowledge_vectors).flatten()
    max_sim_idx = similarities.argmax()
    max_sim_score = similarities[max_sim_idx]
    if max_sim_score > 0.6:
        match_item = ADVANCED_KNOWLEDGE[max_sim_idx]
        if match_item.get("dynamic"):
            return match_item["answer"]()
        return match_item["answer"]
    return None

def generate_response(user_input):
    user_input_lower = user_input.lower()
    if user_input_lower.startswith("scrape "):
        url = user_input[len("scrape "):].strip()
        if url:
            scraped_text = rate_limited_get_url_content(url)
            if scraped_text.startswith("Error") or scraped_text.startswith("Scraping not permitted"):
                return scraped_text
            else:
                preview = scraped_text[:500] + ('...' if len(scraped_text) > 500 else '')
                return f"Successfully scraped content from {url}:<br><pre>{preview}</pre>"
        else:
            return "Please provide a URL to scrape."
    match = custom_semantic_match(user_input)
    if match:
        return match
    sents = sent_tokenize(user_input)
    if sents:
        return f"I'm not sure how to respond to that. You said: \"{sents[0]}\""
    return "I'm not sure how to respond to that."

@app.route("/", methods=["GET", "POST"])
def chat():
    init_db()
    db = get_db()
    chat_history = db.execute("SELECT timestamp, user_input, bot_response FROM conversation ORDER BY id ASC").fetchall()
    response = None

    if request.method == "POST":
        user_input = request.form.get("message", "")
        response = generate_response(user_input)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        db.execute(
            "INSERT INTO conversation (timestamp, user, user_input, bot_response) VALUES (?, ?, ?, ?)",
            (timestamp, "guest", user_input, response)
        )
        db.commit()
        return redirect(url_for("chat"))

    return render_template("chatbot.html", chat_history=chat_history)

if __name__ == "__main__":
    if not os.path.exists(app.config['DATABASE']):
        with app.app_context():
            init_db()
    app.run(debug=True)
