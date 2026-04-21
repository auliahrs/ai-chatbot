# AI Customer Support Chatbot

A simple NLP-based chatbot built using Flask and scikit-learn.

## Features
- Accepts user questions from a web interface
- Matches user input to the most relevant FAQ question
- Returns the most suitable predefined answer

## Tech Stack
- Python
- Flask
- scikit-learn
- HTML
- CSS
- JavaScript

## How it works
The chatbot uses TF-IDF vectorization and cosine similarity to compare user input with stored FAQ questions and returns the closest matching answer.

## How to run
1. Create virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Run `python app.py`
4. Open `http://127.0.0.1:5000`