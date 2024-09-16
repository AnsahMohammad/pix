import json
import os
import re
import requests
import time


class Pixie:
    def __init__(self, model_name="mx-v1", timeout=100, max_tokens=5000):
        self._initialize_model(model_name)
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _initialize_model(self, model_name):
        self.model_name = model_name
        models = {
            "mx-v1": "mistralai/Mistral-7B-Instruct-v0.1/v1/chat/completions",
            "mx-v2": "mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        }
        self.model = models[self.model_name]
        self.model_url = "https://api-inference.huggingface.co/models/" + self.model

    def prompt_generator(self, destination, days, budget, diet, interests, comments):
        prompt = f"Generate travel itinerary to {destination} with timings"
        if days:
            prompt += f" for {days} days"
        # if budget:
        #     prompt += f" with a budget of {budget}"
        if diet:
            prompt += f" with a diet of {diet}"
        if interests:
            prompt += f" with interests in {', '.join(interests)}"        
        if comments:
            prompt += f". Additionally, {comments}"

        return prompt

    def ask_pixie(self, query):
        try:
            response, status_code = self._get_response(query)
            if status_code != 200:
                response = "Sorry, I am not able to process your query at the moment."

            response = self._extract_json(response)

        except Exception as e:
            print(f"Error while processing query: {e}")
            response = "Sorry, I am not able to process your query at the moment"

        return response

    def _get_response(self, query):
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": os.environ.get(
                        "AI_INIT_PROMPT", "Generate travel itinerary for the query"
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 5000,
            "stream": False,
        }
        token = os.environ.get("AI_TOKEN", "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(self.model_url, headers=headers, json=data, timeout=100)
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

    def _extract_json(self, text):
        if not isinstance(text, str):
            print(f"Expected string for text, but got {type(text).__name__}")
            return []

        pattern = r"{.*?}"
        matches = re.findall(pattern, text, re.DOTALL)
        json_content = []
        for match in matches:
            try:
                json_content.append(json.loads(match))
            except json.JSONDecodeError:
                pass
        return json_content


def main():
    pix = Pixie()

    while True:
        query = input("Ask pixie: ")

        start = time.time()
        response = pix.ask_pixie(query)
        end = time.time()

        print("Time taken: ", end - start)

        for res in response:
            print(res)

        print("-" * 50)


if __name__ == "__main__":
    main()