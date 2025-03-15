# Glitch Token Discovery Framework
A modular test-driven framework for detecting and analyzing glitch tokens
## Features 🎯
✅ Glitch Token extraction for an arbitrary model
✅ Modular Test Framework
✅ Fine-tunable
✅ Run your model locally or via API
## How to set up
When using the default tests that come with this repository, only the `GlitchTokenDiscovery.py` file is needed. Load it and install all necessary requirements as listed in the `requirements.txt`file. Also make sure Ollama is set up and the model to be tested is downloaded.
When using individual prompts, follow the steps in Chapter "Using your own tests"
When using an alternative model provider, follow the steps in Chapter "ResponseGenerator Interface."

## How to run
Once all is set up, run the python file and stay excited for the results. Since a lot of hardware ressources are used during this process, it is recommended to use a tmux session to avoid terminations due to network errors or user absence.

## Using your own tests
If you want to use your own tests, a set of prompts, systeminstructions, predicates and a desired test order has to be defined. The following steps will guide through the individual steps.
### Predicates?
Predicates are used to evaluate the test cases. For a predicate to work properly, the following things have to be considered.
- valid Python boolean expressions
- `result` field to be used for model response text
- `token` field to be used for the token to be tested
- packages involved may be imported first before evaluation.

Example predicates:

| Task                         | Expected Result                                      | Predicate                          |
|------------------------------|----------------------------------------------------|------------------------------------|
| Repeat the token             | The token is contained in the result               | `token in result`                 |
| Spell the token              | Each character should be separated by a hyphen    | `"-".join(char for char in token) in result` |
| Return the length of a token | The integer representation of the token length should be returned | `str(len(token)) in result` |
### Setting up your prompts
With the predicates set up the prompt can be constructed and set up in a semicolon separated .CSV file. The structure should be as follows.
- `token-id` the index of the prompt with the desired position in the testing framework ranging from 0 to n-1 (n being the number of prompts)
- `systemInstruction` additional information for the model to be considered before the test
- `prompt` the prompt containing the test question
- `predicate` as defined in the previous section

With the .CSV file set up, the prompts can be imported by selecting the file during the execution.

## Tokenizers
a standard tokenizer.json file of a model can be used. In case a tokenizer file is not available, an adequate .CSV representation of the token vocabulary is sufficient. For that, the file needs to have the following structure: `token-id`;`token`. This file can then either be inserted in the code directly, or chosen by executing the `__main__` method.

## Using a different model provider
If using an alternative API or local model framework such as DeepSeek, OpenAI, Transformers library, this provider can be inserted by implementing the GenerateResult interface and inserting it in the `generator` field of the `GlitchTest` function.
```python
def generateResponse(self, model: str, prompt: str, systemInstruction: str) -> str:
    pass
```
Examples of Implementations ready to use are listed in the Generators package. (DeepSeek, OpenAI)

## Analysing Results
The result will contain a table (.CSV, ";" separated) in which the token and the according token id to each discovered glitch token is listed. Additionally the results of all four tests for this particular token is displayed in the columns on the right. The results could then be evaluated to get a better understanding of the origin and potential patterns the glitch tokens are exhibiting.

## Examples and Tutorials
Four examples are provided in the `Examples` folder.
These examples demonstrate different modular aspects of the algorithm. Custom test cases, intermediate result saving and different model providers are presented in these example files.
