"""
Author: Maximilian Stefan Schreber
email: max.schreber@tum.de
"""

from abc import ABC, abstractmethod
import signal, requests, json, datetime, csv, socket
from datetime import datetime
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm

#### API KEY CONFIGURATION
config = dotenv_values(".env")


####  RESPONSEGENERATOR INTERFACE --------------------------------------------------------------------------------------
class ResponseGenerator(ABC):
    """
    Interface class for generating responses for a LLM request. This interface will be used to do systematic requests
    for experimental purposes.
    """

    @abstractmethod
    def generateResponse(self, model: str, prompt: str, systemInstruction: str) -> str:
        """
        A response from a LLM is created by the given prompt and model descriptions. Afterward the response is
        returned in a string
        :param model: model name as listed in the generator
        :param prompt: prompt to be processed by the LLM (generator)
        :param systemInstruction: Instructions to the model before processing the prompt..Further instructions on how to
        process the prompt
        :return: String of the model response
        """
        pass

#### TIMEOUT HANDLING --------------------------------------------------------------------------------------------------
class TimeoutException(Exception):
    """
    Exception to be raised when a timeout of the generator occurs.
    """
    pass

def timeout_handler(signum, frame):
    # Signal handler for raising TimeoutExceptions in case the model response exceeds a set threshold
    raise TimeoutException("Timeout for model response")

#### OLLAMA IMPLEMENTATION OF RESPONSE GENERATOR INTERFACE -------------------------------------------------------------
class OllamaResponseGenerator(ResponseGenerator):
    TIMEOUT_SECONDS: int = 30  # default timeout duration 30 sec
    API_URL: str = "http://localhost:11434/api/generate"  # default ollama local server api url

    def __init__(self, timeout_seconds: int = None, api_url: str = None, temperature: int = 0):
        # optional change of timeout threshold
        self.TIMEOUT_SECONDS = timeout_seconds if timeout_seconds is not None else self.TIMEOUT_SECONDS
        # optional change of api_url in case of request redirection (i.e. ngrok/Google Colab)
        self.API_URL = api_url if api_url is not None else self.API_URL
        # temperature of model responses. Default value 0 to avoid model misbehavior
        self.temperature = temperature

    def generateResponse(self, model: str, prompt: str, systemInstruction: str) -> str:
        """
        Implementation of ResponseGenerator Interface.
        :param model: model name
        :param prompt: prompt to be processed by the LLM (generator)
        :param systemInstruction: Instructions to the model before processing the prompt..Further instructions on how to
        :return: string of the model response
        """
        # Input data type validation
        if any(not isinstance(var, str) for var in [model, prompt, systemInstruction]):
            raise ValueError("All inputs should be strings.")

        # create json payload for model request
        data = {
            "model": model,
            "system": systemInstruction,
            "prompt": prompt,
            "stream": False,  # response should be sent as a single entity
            "options": {
                "temperature": self.temperature
            },
        }

        # configure the timeout signal handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.TIMEOUT_SECONDS)

        # POST request to ollama API
        try:
            response = requests.post(self.API_URL, json=data)
            signal.alarm(0)
        except TimeoutException as e:
            print("timeout")
            return "timeout"

        # retrieve ollama response and extracting response text
        if response.status_code == 200:
            try:
                output = response.json()
                response_text = output.get("response")  # filter the prompt answer
                if response_text:
                    return response_text
                else:
                    print("No answer")
            except json.JSONDecodeError as error:
                print(f"JSON Decode Error: {error}")
                # return rest of API response
                print("Response Text:", response.text)
        else:  # in case response is not 200 OK
            print(f"Request failed with status code {response.status_code}")
            print("Response Text:", response.text)
        # returning the string response 'ERROR' instead of raising an exception.
        # Necessary for keeping a flow in testing algorithm
        return "ERROR"

#### PUSH NOTIFICATIONS FOR REMOTE STATUS UPDATE (OPTIONAL, ONLY USED AS A CONVENIENCE BENEFIT) ------------------------
class PushNotification:
    @staticmethod
    def send_push(message: str) -> None:
        api_token: str = config.get("PUSHOVER_API_TOKEN")
        user_key: str = config.get("PUSHOVER_USER_KEY")
        if api_token is None or user_key is None: return
        url = "https://api.pushover.net/1/messages.json"
        data = {
            "token": api_token,
            "user": user_key,
            "message": message
        }
        response = requests.post(url, data=data)

#### GLITCH FINDER METHOD TO IMPLEMENT MAIN FUNCTIONALITY --------------------------------------------------------------
class GlitchFinder:
    @staticmethod
    def GlitchTest(generator: ResponseGenerator,
                   path_to_token_csv_or_json: str,
                   path_to_output_csv: str,
                   model: str,
                   path_to_prompts_csv: str,
                   path_to_intermediate_res_folder: str,
                   saving_interval=300,
                   topN=None,
                   sendSMS: bool = False) -> None:
        """
        Main Method to extract glitch tokens from a closed source model with the use of a predefined list of prompts
        listed in this python file.

        :param topN: for testing and debugging of data: if set. the token will be sliced at the set row number
        :param path_to_intermediate_res_folder: folder path to save the intermediate results
        :param path_to_prompts_csv: path to SEMICOLON SEPARATED Prompt CSV of the form
        [PROMPT_ID, SYSTEM_INSTRUCTION, PROMPT_TEXT, PREDICATE]
        :param generator: ResponseGenerator implementation to execute model requests
        :param saving_interval: interval for the intermediate saving option
        :param model: name of the model as listed in ollama server
        :param path_to_token_csv_or_json: File path to the csv-file containing the tokens in the following format:
        <Token-id>;<Token>
        :param path_to_output_csv: File path to the result file in which the values will be positioned as follows:
        <Token-ID>;<Token>;<Prompt1_answer>;<Prompt2_answer>;<Prompt3_answer>
        :param sendSMS if an SMS should be sent to update on status.
        :return results only in form of a saved csv file
        """

        # for csv saving
        def save_token_map_to_csv(data: pd.DataFrame, day, month, year, hour, minute, prompt_nr, folder, model):
            filename = f"intermediate_res_{model}_{day}-{month}-{year}_{hour}{minute}_p{prompt_nr}.csv"
            data.to_csv(f"{folder}/{filename}", sep=";", index=False)

        # Console output coloring
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        bar_format = f"{GREEN}{{l_bar}}{{bar}}{{r_bar}}{RESET}"  # progress bar coloring

        # fixating time and date values for traceability
        now = datetime.now()
        day_now, month_now, year_now, hour_now, min_now = now.day, now.month, now.year, now.hour, now.minute

        # 1 Read in the token map
        # 1.1 Convert JSON-Tokenizer
        print("reading in tokens...")
        if path_to_token_csv_or_json.endswith(".csv"):
            with open(path_to_token_csv_or_json, 'r', encoding='utf-8') as token_csv:
                csv_reader = csv.reader(token_csv, delimiter=';')
                next(csv_reader)  # skip header row
                # create a dictionary with KV-pairs TOKEN-ID:int,TOKEN:str
                token_map: dict[int, str] = {int(row[0]): row[1] for row in csv_reader}
        elif path_to_token_csv_or_json.endswith(".json"):
            with open(path_to_token_csv_or_json, 'r', encoding='utf-8') as token_json:
                json_reader = json.load(token_json)
            vocab = json_reader.get("model", {}).get("vocab", {})
            token_map: dict[int, str] = {token_id: token for token, token_id in vocab.items()}  # swap ID and TOKEN pos
        else:  # wrong file format
            raise ValueError("Wrong file format. Expected .csv ([TOKEN_ID; TOKEN]) "
                             "or .json ({\"model\":{\"vocab\"{\"<token>\": <id>}}})")

        # topN option: slicing the token set to only test upper n tokens of the tokenizer
        if topN is not None:
            old_size_of_map = len(token_map)
            token_map = dict(list(token_map.items())[:topN])
            print(f"sliced Token set from {old_size_of_map} to {topN} tokens.")

        print("reading in prompts...")

        # 2 Read in the prompts, save as nested lists via pandas
        df = pd.read_csv(path_to_prompts_csv, delimiter=";")
        prompts = df.values.tolist()

        # push notification with initialization information
        sendSMS and PushNotification.send_push(
            f"Starting the tests with {len(prompts)} prompts on model {model} with {len(token_map)} "
            f"tokens.\nThe program is running on {socket.gethostname()}.")
        print("starting the testing proces...")
        SMS_count = 0  # for SMS tracing

        # token set initialization
        remaining_tokens = [[token_id, token] for token_id, token in token_map.items()]

        # 3 Iterating every prompt
        for prompt in prompts:
            # extracting prompt parameters
            prompt_index, system_instruction = prompt[0], prompt[1]
            prompt_string, prompt_predicate = prompt[2], prompt[3]
            """
            Lingo for predicate is a string containing the predicate with 'result' as model response
            and 'token' as the token to be tested in the current iteration
            """
            # traceability measures
            print(f"Testing prompt {prompt_index + 1} of {len(prompts)}: {prompt_string}")
            print(f"{RED}{len(remaining_tokens)}{RESET} Tokens remaining")
            # quarter marks for push notifications
            marks: list[int] = [int(len(remaining_tokens) * 0.25), int(len(remaining_tokens) * 0.5),
                                int(len(remaining_tokens) * 0.75), int(len(remaining_tokens))]
            # initialisation of set for tokens to fail the test
            result_tokens = []
            # progress bar and iterable configuration with tqdm progress bar
            pbar = tqdm(remaining_tokens, desc="Processing Tokens ", bar_format=bar_format)
            # 4 iterating every remaining token
            for token_row in pbar:  # token row is a list of values for the token (id, token, predicate,..)
                token, token_index = token_row[1], token_row[0]
                # update progress bar
                pbar.set_description(f"Processing Tokens Stage {prompt_index + 1} '{token}'")
                # 5 sending the request to the response generator
                try:
                    final_prompt = prompt_string.replace("{}", token)  # insert the token
                    result = generator.generateResponse(model, final_prompt,
                                                        system_instruction)
                except Exception as e:
                    result = f"ERROR occurred: {e}"
                    # send push to fathom error origin
                    sendSMS and PushNotification.send_push(f"⚠️ An error occurred. Message: {e}")

                # 6 result evaluation based on provided predicate string
                try:
                    test_eval: bool = bool(eval(prompt_predicate))
                except Exception as e:
                    test_eval: bool = False  # any test failures will mark the test as failed
                    print(
                        f"Error while evaluating predicate for prompt {prompt_index + 1}. Predicate: {prompt_predicate}; Message: {e}")

                if not test_eval:
                    # if eval fails, the token remains in the list for next prompt tests.
                    # nested list row format [id,token,res1,res2,res3,res4,...]
                    result_entry = [token_index, token]
                    if prompt_index != 0:  # first prompt iteration has no result so skipping the result fishing
                        # retrieves the previous results of earlier tests and appends current results for token
                        for res_i in range(2, prompt_index + 2):
                            result_entry.append(token_row[res_i])  # previous results
                    result_entry.append(result)  # new result
                    result_tokens.append(result_entry)  # assembled row into result token list

                # intermediate result saving based on token-id and set interval
                if token_index % saving_interval == 0 or token_index == len(remaining_tokens) - 1:
                    save_token_map_to_csv(pd.DataFrame(result_tokens), day_now, month_now, year_now, hour_now, min_now,
                                          prompt_index, path_to_intermediate_res_folder, model)

                # send status SMS
                if sendSMS and SMS_count in marks:
                    PushNotification.send_push(
                        f"{model}-test Stage {prompt_index + 1} of {len(prompts)}:"
                        f"{SMS_count / len(remaining_tokens) * 100:.2f}% done.")
                SMS_count += 1

            # 7 reallocate remaining tokens for next iteration
            remaining_tokens = result_tokens

            # saving the final results of prompt test
            save_token_map_to_csv(pd.DataFrame(result_tokens), day_now, month_now, year_now, hour_now, min_now,
                                  prompt_index, path_to_intermediate_res_folder,
                                  f"{model}_finalresultIn{prompt_index}.csv")

        # final save of result of last prompt test iteration
        print("saving the final results...")
        columns = ["token_id", "token"]
        for prompt_index in range(len(prompts)):  # creating column names for all prompts
            columns.append(f"res_{prompt_index + 1}")  # enumerating all result columns
        end_result = pd.DataFrame(columns=columns, data=remaining_tokens)

        # csv export
        end_result.to_csv(path_to_output_csv, index=False, sep=";")

        # end communication
        print(f"{RED}{len(remaining_tokens)}{RESET} Tokens failed all tests and will be saved in a final csv-file.")

        sendSMS and PushNotification.send_push(f"{len(remaining_tokens)} tokens found in test for model {model}."
                                               f"The files are saved in {path_to_output_csv}.🥳 ")
        print(f"file saved in {BLUE}{path_to_output_csv}{RESET}")


#### MAIN METHODOLOGY --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    The main method is configured to choose all necessary files by file select windows 

    """
    print("Choosing OLLAMA as response generator as default..\n")

    # Import model name
    model_name = input("Enter model name as listed in ollama: ")

    # Import Tokenizer
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    tokenizer_file_path = filedialog.askopenfilename(
        title="Select Tokenizer file CSV or JSON",
        filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")]
    )

    # Import Intermediate result folder path
    intermediate_folder_path = filedialog.askdirectory(title="Choose the Folder to save intermediate results in.")

    # Import Filepath of Result file
    result_file_path = filedialog.asksaveasfilename(
        title="Choose the folder to save final results in",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )

    # Import Prompts
    prompt_file_path = filedialog.askopenfilename(
        title="Choose the CSV file containing the model prompts",
        filetypes=[("CSV files", "*.csv")]
    )

    # Import parameters into GlitchTestMethod
    GlitchFinder.GlitchTest(
        path_to_token_csv_or_json=tokenizer_file_path,
        path_to_intermediate_res_folder=intermediate_folder_path,
        path_to_output_csv=result_file_path,
        path_to_prompts_csv=prompt_file_path,
        model=model_name,
        generator=OllamaResponseGenerator()
    )