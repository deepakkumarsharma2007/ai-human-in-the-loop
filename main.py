from dotenv import load_dotenv

from genai_core.rag.script_to_embed import embed


def main():
    print("Hello from ai-human-in-the-loop!")

    load_dotenv()
    embed() #embedding is done


if __name__ == "__main__":
    main()
