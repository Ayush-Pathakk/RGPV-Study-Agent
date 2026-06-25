from flask import Flask, render_template, request, jsonify
from query import load_index, ask

app = Flask(__name__)

# Load index once when server starts — stays in memory for all requests.
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

    answer = ask(index, question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    # use_reloader=False prevents Flask from loading the index twice on startup.
    app.run(debug=True, use_reloader=False)