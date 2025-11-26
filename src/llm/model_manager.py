import os
from typing import Any

import litellm
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_litellm import ChatLiteLLM
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    EMBEDDING_MODELS_CONFIG,
    GEMINI,
    LLM_COMMON_PARAMETERS,
    MODELS_CONFIG,
    OLLAMA,
    OPENAI,
)
from src.utils.logger import logger


class LLMModelManager:
    def __init__(self):
        load_dotenv()
        self._setup_environment()
        self.model_alias = os.getenv("MODEL_NAME") or DEFAULT_MODEL
        self.embedding_model_alias = os.getenv("EMBEDDING_MODEL_NAME") or DEFAULT_EMBEDDING_MODEL

    def _setup_environment(self):
        """Setup litellm configuration."""
        litellm.success_callback = [self._log_success]
        litellm.failure_callback = [self._log_failure]

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """Get chat completion from the specified model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the model

        Returns:
            Completion response from the model
        """
        params = self._prepare_model_params(MODELS_CONFIG, self.model_alias)
        params.update(kwargs)
        return litellm.completion(
            messages=messages,
            drop_params=True,
            **params,
        )

    def get_chat_model(self, **kwargs) -> ChatLiteLLM:
        """Get a chat model instance based on the configured provider.

        Args:
            **kwargs: Additional parameters to override the default configuration.

        Returns:
            An instance of the appropriate chat model class.
        """
        params = self._prepare_model_params(MODELS_CONFIG, self.model_alias)
        params.update(kwargs)
        return ChatLiteLLM(**params)

    def get_embedding_model(
        self, **kwargs,
    ) -> OpenAIEmbeddings | GoogleGenerativeAIEmbeddings | OllamaEmbeddings:
        """Get an embedding model instance based on the configured provider.

        Args:
            **kwargs: Additional parameters to override the default configuration.

        Returns:
            An instance of the appropriate embedding model class.

        Raises:
            ValueError: If the provider is not supported.
        """
        params = self._prepare_model_params(EMBEDDING_MODELS_CONFIG, self.embedding_model_alias)
        params.update(kwargs)
        provider = params.get("provider")

        if not provider:
            raise ValueError("Provider not specified in model configuration")

        provider = provider.lower()

        if provider == str(OPENAI).lower():
            return OpenAIEmbeddings(**params)
        if provider == str(GEMINI).lower():
            params["google_api_key"] = os.environ["GEMINI_API_KEY"]
            return GoogleGenerativeAIEmbeddings(**params)
        if provider == str(OLLAMA).lower():
            return OllamaEmbeddings(
                model=params.get("model"),
                base_url=params.get("api_base"),
            )
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Must be one of: {OPENAI.lower()}, {GEMINI.lower()}, {OLLAMA.lower()}",
        )

    def _prepare_model_params(self, model_configs: list[dict], model_alias: str) -> dict:
        """Prepare model parameters by merging common parameters with model-specific config.

        Args:
            model_configs: List of model configurations to search through.
            model_alias: The alias of the model to find configuration for.

        Returns:
            Dictionary of merged parameters.
        """
        params = LLM_COMMON_PARAMETERS.copy()

        for model_config in model_configs:
            if model_config["model_alias"] == model_alias:
                params.update(model_config)
                break

        return params

    def _log_success(self, kwargs, _, start_time, end_time):
        """Callback for successful API calls."""
        logger.info(
            "LiteLLM API call successful - Call ID: %s, Duration: %s",
            kwargs.get("litellm_call_id"),
            end_time - start_time,
        )

    def _log_failure(self, _, error_response, start_time, end_time):
        """Callback for failed API calls."""
        logger.error(
            "LiteLLM API call failed - Error: %s, Duration: %s",
            error_response,
            end_time - start_time,
        )
