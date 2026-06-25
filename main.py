from query import load_index, ask


def main():


    # Load index once — model stays in memory for the entire session.
    index = load_index()
    print("=" * 50)
    print("RGPV Study Assistant")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)
    print("\nReady. Ask your question:\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye.")
            break

        print("\nAssistant: ", end="", flush=True)
        answer = ask(index, question)
        print(answer)
        print()


if __name__ == "__main__":
    main()