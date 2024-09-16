import time
import requests
import json
import os


def ask_pixie(query):
    response, status_code = process_query(query)
    if status_code != 200:
        response = "Sorry, I am not able to process your query at the moment."

    return response


def process_query(query, model_name="mx-v1"):
    models = {
        "mx-v1": "mistralai/Mistral-7B-Instruct-v0.1/v1/chat/completions",
        "mx-v2": "mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
    }
    model = models[model_name]
    model_url = "https://api-inference.huggingface.co/models/" + model

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": os.environ.get(
                    "AI_INIT_PROMPT", "Generate travel itirinary for the query"
                ),
            },
            {"role": "user", "content": query},
        ],
        "max_tokens": 5000,
        "stream": False,
    }
    token = os.environ.get("AI_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        response = requests.post(model_url, headers=headers, json=data, timeout=100)
        response_json = response.json()
        generated_text = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", None)
        )

    except Exception as e:
        print(f"Error while sending request to AI model: {e}")
        return None, 500

    return generated_text, 200


def main():
    while True:
        query = input("Ask pixie: ")

        # query processing : action or question

        start = time.time()
        response = ask_pixie(query)
        end = time.time()

        # post-answer processing : performing actions, or replying question

        print("Time taken: ", end - start)
        print("pixie: ", response)


if __name__ == "__main__":
    main()
