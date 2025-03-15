"""
Author: Maximilian Stefan Schreber
Date: 15.03.2025

This example demonstrates how to insert custom test cases into this framework. For demonstration purposes, an alternative
OpenAI generator implementation is used.
Note that an actual OpenAI API Key needs to be obtained and inserted in the
DeepSeekResponseGenerator interface to work. Also notice that this example uses the DeepSeek tokenizer since there
currently is no publicly available tokenizer file available for GPT-4o. This example could be understood as a reference
test to compare the outcomes of DeepSeek-v3 and GPT-4o.
"""
from Generators import GPTResponseGenerator as GPT
from GlitchTokenDiscovery import GlitchFinder

GlitchFinder.GlitchTest(
    generator = GPT.GPTResponseGenerator(), # OpenAI API as model provider
    path_to_token_csv_or_json= "tokenizer_deepseek-v3.json", # Tokenizer
    path_to_output_csv="example4_results.csv", # Output
    model="GPT-4o", # Model
    path_to_prompts_csv = "custom_prompts.csv" # The custom prompts are inserted in the additional .csv file.
)