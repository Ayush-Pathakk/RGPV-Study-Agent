from flask import Flask, render_template, request, jsonify, session
from query import load_index, ask, summarize_exchange
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

print("Loading index...")
index = load_index()
print("Index ready.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data     = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    memory = session.get("memory", "")
    answer = ask(index, question, memory)
    session["memory"] = summarize_exchange(question, answer)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False)