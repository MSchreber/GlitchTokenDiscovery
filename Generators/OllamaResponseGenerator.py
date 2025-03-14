"""
Author: Maximilian Stefan Schreber
Date: 13.03.2025
"""
from Generators import ResponseGenerator
import requests, json, signal

def timeout_handler(signum, frame):
    """
    standard time-out handler
    """
    raise TimeoutException('Timed out while waiting for model response.')

class TimeoutException(Exception):
    """
    Timeout exception to be thrown when a timeout occurs.
    """
    pass

class OllamaResponseGenerator(ResponseGenerator):
    @staticmethod
    def generateResponse(self,model:str, prompt:str, system_instructions:str) -> str:
        """
        Program connector to use Ollama as a model provider. Given Inputs result in the model respose text.
        :param model: model name as listed in the Ollama service
        :param prompt: prompt text
        :param system_instructions: instruction text
        :return: response text
        """

        timeout = 20 # fixed timeout parameter for execution
        if not isinstance(prompt, str):
            raise ValueError("Wrong Input Type :(",
                             "The Prompt should be a String, change the input type or add Diaresises")

        api_url = "http://localhost:11434/api/generate" # local api, could be changed to use a sevice such as ngrok.
        temperature = 0 # fixed to not provoke any unexpected behavior
        # create data set for prompt
        data = {
            "system": system_instructions,
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            },
        }
        # send a REST POST request
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            response = requests.post(api_url, json=data)
            signal.alarm(0) # timeout alarm
        except TimeoutException as e:
            print("timeout")
            return "timeout"

        # retrieve response
        if response.status_code == 200:
            try:
                output = response.json()
                response_text = output.get("response")  # filter the prompt answer
                if response_text:
                    return response_text
                else:
                    print("No answer from the model.")
            except json.JSONDecodeError as error:
                print(f"JSON Decode Error: {error}")
                # return rest of API response
                print("Response Text:", response.text)
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response Text:", response.text)
        return f"ERROR {response.status_code}, RESPONSE TEXT:{response.text}"
