"""
Author: Maximilian Stefan Schreber
Date: 13.03.2025
"""

from Generators import ResponseGenerator
from openai import OpenAI
client = OpenAI(api_key="<key>", base_url="https://api.deepseek.com")

class DeepSeekResponseGenerator(ResponseGenerator):
    """
    Implementation of the GenerateResponse interface to make DeepSeek Accessible
    """
    @staticmethod
    def generateResponse(self,model:str, prompt:str,systemRole:str) -> str:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": systemRole},
                {"role": "user", "content": prompt},
            ],
            stream=False # forcing the model to return the respnse as a whole
        )
        return response.choices[0].message.content
