import json
import os
import re
import requests
import spacy
import time
import requests
from bs4 import BeautifulSoup


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
        prompt = f"Generate travel itinerary to {destination}"
        if days:
            prompt += f" for {days+1} days"
        # if budget:
            # prompt += f" with a budget of {budget}"
        if diet:
            prompt += f" with a diet of {diet}"
        if interests:
            prompt += f" with interests in {', '.join(interests)}"
        if comments:
            prompt += f". Additionally, {comments}"

        return prompt

    def ask_pixie(self, query):
        entities = []

        try:
            response, status_code = self._get_response(query)
            if status_code != 200:
                response = "Sorry, I am not able to process your query at the moment."

            response = self._extract_json(response)

            for res in response:
                day = []
                for value in res.values():
                    day.append(self.extract_entities(value))
                entities.append(day)

            entities = self.process_entities(entities)

        except Exception as e:
            print(f"Error while processing query: {e}")
            response = "Sorry, I am not able to process your query at the moment"

        return response, entities

    def extract_entities(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def process_entities(self, entities):
        processed_entities = []
        for day in entities:
            activities = [activity for sublist in day for activity in sublist]
            processed_entities.append(activities)

        organized_entities = self.organize_entities(processed_entities)

        return organized_entities

    def organize_entities(self, processed_entities):
        organized_entities = []
        for day in processed_entities:
            organized_entities.append(self._organize_entity(day))

        return organized_entities

    def fetch_image(self, query):
        """
        Fetches the main image URL from the first Wikipedia search result.

        Args:
        - query (str): Search query.

        Returns:
        - str: Image URL or error message.
        """

        # Wikipedia search URL and parameters
        search_url = "https://en.wikipedia.org/w/index.php"
        params = {
            "search": query,
            "title": "Special:Search",
            "profile": "default",
            "fulltext": 1,
            "ns0": 1,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }

        try:
            response = requests.get(
                search_url, params=params, headers=headers, timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            first_result = soup.find("div", class_="mw-search-result-heading")

            if not first_result:
                return "No search results found."

            # Extract first result link
            first_link = first_result.find("a")["href"]
            page_url = f"https://en.wikipedia.org{first_link}"

            # Send request to first result page
            page_response = requests.get(page_url, headers=headers, timeout=10)
            page_response.raise_for_status()

            page_soup = BeautifulSoup(page_response.text, "html.parser")
            main_image = page_soup.select_one("table.infobox img")

            if main_image:
                image_url = f"https:{main_image['src']}"
                return image_url
            else:
                return "No image found on the page."

        except requests.RequestException as e:
            return f"Error fetching image: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

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
            response = requests.post(
                self.model_url, headers=headers, json=data, timeout=100
            )
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

    def _organize_entity(self, entities):
        """
        entities is list of tuples eg : [('the Grand Canyon', 'LOC')]
        (per day)
        """
        organized_entity = {"GPE": [], "LOC": [], "ORG": [], "images": []}

        for obj, entity in entities:
            if entity in organized_entity:
                organized_entity[entity].append(obj)
                organized_entity["images"].append(self.fetch_image(obj))

        return organized_entity


def main():
    pix = Pixie()

    while True:
        query = input("Ask pixie: ")

        start = time.time()
        response, entities = pix.ask_pixie(query)
        end = time.time()

        print("Processing time : ", end - start)

        print("Response : ")
        for res in response:
            print(res)

        print("Entities : ", entities)


if __name__ == "__main__":
    main()
