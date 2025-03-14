"""
Author: Maximilian Stefan Schreber
Date: 13.03.2025
"""

from Generators import ResponseGenerator
from openai import OpenAI

class GPTResponseGenerator(ResponseGenerator):
    @staticmethod
    def generateResponse(self,model:str, prompt:str,systemInstructions:str) -> str:

        client = OpenAI(api_key="<key>")

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemInstructions},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        return str(completion.choices[0].message.content)