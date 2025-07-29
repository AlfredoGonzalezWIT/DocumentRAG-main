# embedding_helper.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

class EmbeddingHelper:
    def __init__(self, env_file: str = "local.env"):
        """
        Initializes the EmbeddingHelper by loading environment variables and configuring the Azure OpenAI client.
        """
        load_dotenv(dotenv_path=env_file)

        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://<resource>.openai.azure.com
        self.model = os.getenv("AZURE_OPENAI_ENGINE")   # This must be your DEPLOYMENT NAME
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if not all([self.api_key, self.endpoint, self.model, self.api_version]):
            raise ValueError("Azure OpenAI configuration is incomplete. Please check your local.env file.")

        # Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )

    def get_embedding(self, text: str) -> list:
        """
        Generate an embedding for the given text using Azure OpenAI.

        :param text: The input text to embed.
        :return: A list of floats representing the embedding vector.
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model  # This is your Azure deployment name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
