import requests


class OllamaClient:

    def __init__(self):

        self.base_url = "http://localhost:11434"

        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def embed(
        self,
        text
    ):

        response = requests.post(

            f"{self.base_url}/api/embeddings",

            json={
                "model": self.embed_model,
                "prompt": text
            }

        )

        if response.status_code != 200:

            return None

        return response.json()["embedding"]

    def generate(
        self,
        prompt
    ):

        response = requests.post(

            f"{self.base_url}/api/generate",

            json={
                "model": self.gen_model,
                "prompt": prompt,
                "stream": False
            }

        )

        if response.status_code != 200:

            return "Generation failed"

        return response.json()["response"]