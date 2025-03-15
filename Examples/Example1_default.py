"""
Author: Maximilian Stefan Schreber
Date: 15.03.2025

This example works by using only the necessary inputs.
1. the tokenizer (here as json from Llama2:7b. Inspect Notice.txt for licencing and open source usage)
2. the output path
3. the model name to be tested
"""
from GlitchTokenDiscovery import GlitchFinder #Import the GlitchFinder class

GlitchFinder.GlitchTest(
    path_to_token_csv_or_json= "tokenizer_llama2-7b.json", # Tokenizer
    path_to_output_csv="example1_results.csv", # Output
    model="llama2:7b" # Model
)
