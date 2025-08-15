from src.llm.model_manager import LLMModelManager
from litellm import completion
from pydantic import BaseModel
import json
import os

def main():
    llm_manager = LLMModelManager()
    # Example chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]

    response = llm_manager.chat_completion(messages)
    print("\nChat Response:")
    print(response['choices'][0]['message']['content'])


if __name__ == "__main__":
    # Run using this comamnd
    # PYTHONPATH=/Users/deepthir/projects/code/tech_radar_chat python3 examples/llm_usage_example.py
    main()
