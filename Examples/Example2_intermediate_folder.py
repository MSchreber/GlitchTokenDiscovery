"""
Author: Maximilian Stefan Schreber
Date: 15.03.2025

This example works by using the necessary inputs and a path field for saving intermediate results.
1. the tokenizer (here as json from Llama2:7b. Inspect Notice.txt for licencing and open source usage)
2. the output path
3. the model name to be tested
4. path to the intermediate folder
5. custom saving interval (here 500 tests)
"""
from GlitchTokenDiscovery import GlitchFinder #Import the GlitchFinder class

GlitchFinder.GlitchTest(
    path_to_token_csv_or_json = "tokenizer_llama2-7b.json", # Tokenizer
    path_to_output_csv = "example2_results.csv", # Output
    model = "llama2:7b", # Model
    path_to_intermediate_res_folder = "Intermediate_Results", # Intermediate Folder name
    saving_interval = 500 # Save the results every 500 tests
)
