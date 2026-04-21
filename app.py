from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

with open("data/faq.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


def get_best_answer(user_message, threshold=0.2):
    user_vector = vectorizer.transform([user_message])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0, best_match_index]

    if best_score < threshold:
        return "Sorry, I could not find a matching answer. Please contact customer support for more help."

    return answers[best_match_index]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please enter a message."}), 400

        reply = get_best_answer(user_message)
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)