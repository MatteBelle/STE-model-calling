"""
Functions for loading and handling datasets.
"""

import json
from typing import List, Dict, Tuple

def load_dataset(filepath: str) -> List[Dict[str, str]]:
    """
    Load the dataset from a JSON file.
    
    Args:
        filepath: Path to the dataset JSON file
        
    Returns:
        List of query dictionaries
    """
    print(f"Loading dataset from {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return []

def load_api_metadata() -> Tuple[Dict[str, str], List[str]]:
    """
    Load API descriptions and API list.
    
    Returns:
        Tuple of (API_descriptions, API_list)
    """
    with open("STE/test_compare/utility/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)
    
    with open("STE/test_compare/utility/API_list.json", "r", encoding='utf-8') as f:
        API_list = json.load(f)
        
    return API_descriptions, API_list