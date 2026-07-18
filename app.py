from flask import Flask, render_template, request, jsonify, session
from query import load_index, ask, summarize_exchange
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise SystemExit("FLASK_SECRET_KEY not set in .env")

print("Loading index...")
index = load_index()
print("Index ready.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    memory = session.get("memory", "")
    answer, resolved_question = ask(index, question, memory)
    if resolved_question is not None:
        session["memory"] = summarize_exchange(resolved_question, answer)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True, use_reloader=True)