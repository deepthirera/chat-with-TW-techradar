import os
from typing import Dict, Any
import litellm
from litellm import completion
from config import ( DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL, 
                    MODELS_CONFIG, EMBEDDING_MODELS_CONFIG, 
                    LLM_COMMON_PARAMETERS )
from src.utils.logger import logger
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


class LLMModelManager:
    def __init__(self):
        load_dotenv()
        self._setup_environment()
        self.model_alias = os.getenv('model_name') or DEFAULT_MODEL
        self.embedding_model_alias = os.getenv('embedding_model_name') or DEFAULT_EMBEDDING_MODEL
        
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
        params = self._prepare_chat_model_params()
        print(params)
        params.update(kwargs)
        return completion(
            messages=messages,
            drop_params=True,
            **params
        )
    
    def get_chat_model(self, **kwargs) -> ChatLiteLLM:
        params = self._prepare_chat_model_params()
        params.update(kwargs)
        return ChatLiteLLM(**params)

    def get_embedding_model(self, **kwargs):
        params = self._prepare_embedding_model_params()
        params.update(kwargs)
        provider = params.get("provider")

        match str(provider):
            case "openai":
                return OpenAIEmbeddings(**params)
            case "google":
                return GoogleGenerativeAIEmbeddings(**params)
            case "ollama":
                return OllamaEmbeddings(model=params.get("model"), base_url=params.get("api_base"))
            case _:
                raise ValueError(f"Invalid provider: {provider}")

    def _prepare_chat_model_params(self):
        params = LLM_COMMON_PARAMETERS.copy()

        for model_config in MODELS_CONFIG:
            if model_config['model_alias'] == self.model_alias:
                params.update(model_config)
                break
        return params # handle wrong models mentioned in env
    
    def _prepare_embedding_model_params(self):
        params = LLM_COMMON_PARAMETERS.copy()

        for model_config in EMBEDDING_MODELS_CONFIG:
            if model_config['model_alias'] == self.embedding_model_alias:
                params.update(model_config)
                break
        return params # handle wrong models mentioned in env
    
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
