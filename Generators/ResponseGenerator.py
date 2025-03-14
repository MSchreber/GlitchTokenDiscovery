"""
Author: Maximilian Stefan Schreber
Date: 10.03.2025
"""
from abc import ABC, abstractmethod

class ResponseGenerator(ABC):
    """
    Interface class for generating responses for a LLM request. This interface will be used to do systematic requests
    for experimental purposes.
    """
    @abstractmethod
    def generateResponse(self,model:str, prompt:str, systemInstructions:str) -> str:
        """
        A response from a LLM is created by the given prompt and model descriptions. Afterwards the response is
        returned in a string
        :param systemInstructions: Further Instructions for the model to consider during execution.
        :param model: Model name as listed in the according framework
        :param prompt: Model prompt to be executed
        :return: String model response
        """
        pass