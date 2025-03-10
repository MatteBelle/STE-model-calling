"""
Constants used throughout the project.
"""

# Model settings
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FINETUNED_MODEL_PATH = "../finetuning/outputs/metric_evaluation_assistant/final_model"
TEST_DATASET_PATH = "STE/test_compare/test_dataset.json"
TEMPERATURE = 0.6
MAX_TURNS = 3

# Output settings
DEFAULT_OUTPUT_DIR = "STE/test_compare/outputs"

# System message defaults
DEFAULT_SYSTEM_MSG = "You are a bot that creates and responds to evaluation queries."