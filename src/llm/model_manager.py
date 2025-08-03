from pathlib import Path
import yaml
import os
from typing import Optional, Dict, Any
import litellm
from litellm import completion, embedding
from config.constants import LLM_CONFIG_FILE
from logger import logger
from dotenv import load_dotenv

class LLMModelManager:
    def __init__(self):
        """Initialize the LLM model manager with configuration.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default path.
        """
        config_path = LLM_CONFIG_FILE
        self.config = self._load_config(config_path)
        self._setup_environment()
        
    def _load_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_environment(self):
        """Setup litellm configuration."""
        litellm.success_callback = [self._log_success]
        litellm.failure_callback = [self._log_failure]

    def chat_completion(
        self,
        messages: list[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Get chat completion from the specified model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use. If None, uses default model from config
            **kwargs: Additional parameters to pass to the model
        
        Returns:
            Completion response from the model
        """
        load_dotenv()
        self.model = os.getenv('model') or self.config['default_model']
        params = self.config['common_parameters'].copy()

        for model_config in self.config.get('models'):
            if model_config['model_alias'] == self.model:
                params.update(model_config)
                break

        params.update(kwargs)
        return completion(
            messages=messages,
            drop_params=True,
            **params
        )

    # def get_embedding(
    #     self,
    #     text: list[str],
    #     **kwargs
    # ) -> list[float]:
    #     """Get embeddings for the given text.
        
    #     Args:
    #         text: Text to get embeddings for
    #         model: Model to use. If None, uses default embedding model from config
        
    #     Returns:
    #         List of embedding values
    #     """

    #     self.embedding_model = os.getenv('embedding_model') or self.config['default_embedding_model']
    #     params = self.config['common_parameters'].copy()

    #     for model_config in self.config.get('embeddings'):
    #         if model_config['model_alias'] == self.embedding_model:
    #             params.update(model_config)
    #             break

    #     params.update(kwargs)

    #     response = embedding(
    #         input=text,
    #         drop_params=True,
    #         **params
    #     )
        
    #     return response['data'][0]['embedding']

    def _log_success(self, kwargs, completion_response, start_time, end_time):
        """Callback for successful API calls."""
        logger.info("LITELLM: in success callback function")
        logger.info("kwargs %s", kwargs['litellm_call_id'])
        logger.info("start_time %s", start_time)
        logger.info("end_time %s", end_time)

    def _log_failure(self, kwargs, error_response, start_time, end_time):
        """Callback for failed API calls."""
        logger.error("LITELLM: in failure callback function")
        logger.error("error_response %s", error_response)
        logger.error("start_time %s", start_time)
        logger.error("end_time %s", end_time)
