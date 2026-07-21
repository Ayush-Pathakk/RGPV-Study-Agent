from flask import Flask, render_template, request, jsonify, session
from query import (
    load_index,
    ask,
    summarize_exchange,
    identify_key_provider,
    validate_key,
)
import os
import threading
 
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

if not app.secret_key:
    raise SystemExit("FLASK_SECRET_KEY not set")

# Lazy-loaded global index
_index = None
_index_lock = threading.Lock()


def get_index():
    global _index

    if _index is None:
        with _index_lock:
            if _index is None:
                print("Loading index...")
                _index = load_index()
                print("Index ready.")

    return _index


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/has-api-key")
def has_api_key():
    return jsonify({"has_key": bool(session.get("api_key"))})


@app.route("/set-api-key", methods=["POST"])
def set_api_key():
    key = (request.get_json().get("api_key") or "").strip()

    provider = identify_key_provider(key)
    valid, message = validate_key(key, provider)

    if valid:
        session["api_key"] = key
        session["provider"] = provider

    return jsonify(
        {
            "valid": valid,
            "message": message,
        }
    )


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    api_key = session.get("api_key")
    provider = session.get("provider")

    if not api_key or not provider:
        return jsonify(
            {
                "error": "no_api_key",
                "answer": "Please add your Groq or Gemini API key first.",
            }
        ), 401

    memory = session.get("memory", "")
    topic = session.get("topic", "")

    try:
        index = get_index()

        answer, resolved_question, topic, sources = ask(
            index,
            question,
            api_key,
            provider,
            memory,
            topic,
        )

        session["topic"] = topic

    except Exception as e:
        err = str(e).lower()

        if (
            "authentication" in err
            or "invalid api key" in err
            or "401" in err
            or "permission" in err
        ):
            session.pop("api_key", None)
            session.pop("provider", None)

            return jsonify(
                {
                    "error": "invalid_api_key",
                    "answer": "Your API key was rejected. Please re-enter it.",
                }
            ), 401

        raise

    if resolved_question is not None:
        session["memory"] = summarize_exchange(
            resolved_question,
            answer,
        )

    return jsonify(
        {
            "answer": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )