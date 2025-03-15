"""
Author: Maximilian Stefan Schreber
Date: 15.03.2025

This example demonstrates how to use the DeepSeek API
Note that an actual DeepSeek API Key needs to be obtained and inserted in the
DeepSeekResponseGenerator interface to work.
"""
from Generators import DeepSeekResponseGenerator as deepseek
from GlitchTokenDiscovery import GlitchFinder

GlitchFinder.GlitchTest(
    generator = deepseek.DeepSeekResponseGenerator(), # DeepSeek API as model provider
    path_to_token_csv_or_json= "tokenizer_deepseek-v3.json", # Tokenizer
    path_to_output_csv="example1_results.csv", # Output
    model="deepseek-chat" # Model
)
