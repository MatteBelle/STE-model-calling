# Metric Evaluation Assistant

A fine-tuned model for processing and responding to metric evaluation queries for natural language processing tasks.

## Project Overview

This project adapts the STE (Simulated Trial and Error) framework to build a specialized assistant that processes natural language queries about evaluation metrics like ROUGE, BLEU, BERTScore, etc., and formulates proper metric calls with appropriate parameters.

The system:
1. Asks the chosen model to generate a user query that exploits an extracted subgroup of metrics
2.Asks the model to answer the user query in a structured json format, given metric docs
3. Identifies which metrics are needed based on the query
4. Generates proper Action/Action Input calls with correct parameters
5. Returns formatted responses that can be used programmatically
6. Cycle this process to create a dataset
7. Provide modules for fine-tuning and inference

## Repository Structure

```
STE-model-calling/
├── STE/
│   ├── tool_metadata/
│   │   ├── API_list.json        # List of supported metrics
│   │   └── API_descriptions.json # Detailed descriptions of each metric
│   ├── finetuning/              # Fine-tuning related files
│   │   ├── dataset.json         # Dataset for fine-tuning
│   │   └── run_finetuning.sh    # Script to run fine-tuning
│   ├── test_compare/            # Testing and comparison tools
│   │   ├── utility/             # Utility functions for testing
│   │   └── config/              # Configuration files
│   ├── saved_results/           # Saved evaluation results
│   └── main.py                  # Main STE implementation
├── extract-dataset.py           # Script to extract training data
└── README.md                    # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.2.0+
- Transformers 4.38.0+
- Hugging Face account (for model access)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/mattebelle/STE-model-calling.git
   cd STE-model-calling
   ```

2. Install the required dependencies:
   ```
   pip install -r STE/finetuning/requirements.finetuning.txt
   ```

3. Ensure you have the necessary API keys:
   - For Hugging Face access to Meta-Llama models

## Usage

### Running Evaluation

To test the fine-tuned model against the base model:

```bash
python STE/test_compare/run_comparison.py --dataset TEST_DATASET_PATH
```

### Fine-tuning the Model

The model can be fine-tuned on additional data:

```bash
cd STE/finetuning
./run_finetuning.sh
```

### Creating a New Dataset

To extract and format new training data:

```bash
python extract-dataset.py
```

## The Model

This project fine-tunes meta-llama/Llama-3.1-8B-Instruct to create a specialized evaluation assistant that:

- Understands metric documentation
- Formats parameters correctly
- Handles multiple metric calls in one query
- Provides consistently formatted outputs

## License

This project is modified from original work under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is built upon the STE (Simulated Trial and Error) framework
- Original paper: "LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error"
- Uses the Meta-Llama/Llama-3.1-8B-Instruct model architecture