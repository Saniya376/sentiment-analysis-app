import csv
from flask import send_file
import io

from flask import Flask, request, render_template
import pickle
import sqlite3
import os

app = Flask(__name__)

# Load model
with open('model/sentiment_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

# Initialize database
def insert_data(text, prediction):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sentiments (text, prediction) VALUES (?, ?)", (text, prediction))
    conn.commit()
    conn.close()

def get_all_data():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sentiments")
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        result = prediction
        insert_data(text, prediction)

    data = get_all_data()
    return render_template('index.html', result=result, data=data)

@app.route('/export')
def export():
    ...
    # Your export logic remains here
    return send_file(...)

if __name__ == '__main__':
    print("✅ Flask app is starting...")
    app.run(debug=True)