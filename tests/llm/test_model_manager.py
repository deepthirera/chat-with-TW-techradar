import pytest
from unittest.mock import patch
from src.llm.model_manager import LLMModelManager
import os

@pytest.fixture
def env_vars():
    return {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_API_KEY': 'test-google-key',
        'OLLAMA_HOST': 'http://localhost:11434'
    }

@pytest.fixture
def expected_params_for_model(model_alias):
    model_alias_with_expected_params = {
        'gpt-5-mini': {
            'model_alias': 'gpt-5-mini',
            'model': 'gpt-5-mini-2025-08-07',
            'max_tokens': 4096,
            'temperature': 0.7
        },
        'ollama-mistral': {
            'model_alias': 'ollama-mistral',
            'model': "ollama/mistral",
            'api_base': 'http://localhost:11434',
            'max_tokens': 4096,
            'temperature': 0.7
        }
    }
    return model_alias_with_expected_params[model_alias]

def test_model_default_from_config(env_vars):
    """Test that model defaults to config value when not in env"""
    with patch.dict(os.environ, {'MODEL_NAME': '', 'EMBEDDING_MODEL_NAME': '', **env_vars}, clear=True), \
        patch('src.llm.model_manager.completion', return_value={'choices': [{'message': {'content': 'test-response'}}]}):
        manager = LLMModelManager()
        manager.chat_completion([])
        assert manager.model_alias == 'ollama-mistral'
        assert manager.embedding_model_alias == 'ollama-nominic'

@pytest.mark.parametrize("model_alias", [
    'gpt-5-mini',
    'ollama-mistral'
])
def test_chat_completion_parameters(env_vars, expected_params_for_model, model_alias):
    """Test that chat completion is called with correct parameters"""
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch.dict(os.environ, {'MODEL_NAME': model_alias, **env_vars}), \
        patch('src.llm.model_manager.completion') as mock_completion:
        
        print("**************")
        print(model_alias)
        print(expected_params_for_model)

        manager = LLMModelManager()
        manager.chat_completion(messages)
        assert manager.model_alias == model_alias
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        assert call_args['model'] == expected_params_for_model['model']
        assert call_args['messages'] == messages
        for key, value in expected_params_for_model.items():
            assert call_args[key] == value

@pytest.mark.parametrize("model_alias", [
    'gpt-5-mini',
    'ollama-mistral'
])
def test_get_chat_model_to_return_chat_model_instance(env_vars, expected_params_for_model, model_alias):
    """Test that chat completion is called with correct parameters"""
    
    with patch.dict(os.environ, {'MODEL_NAME': model_alias, **env_vars}), \
        patch('src.llm.model_manager.ChatLiteLLM') as mock_chat_completion:
        
        manager = LLMModelManager()
        manager.get_chat_model()
        mock_chat_completion.assert_called_once()
        call_args = mock_chat_completion.call_args[1]
        assert call_args['model'] == expected_params_for_model['model']
        for key, value in expected_params_for_model.items():
            assert call_args[key] == value

def test_override_parameters_in_completion(env_vars):
    """Test that passed kwargs override default parameters in completion"""
    messages = [{"role": "user", "content": "Hello"}]
    override_params = {
        'temperature': 0.9,
        'max_tokens': 100
    }
    
    with patch.dict(os.environ, {'MODEL_NAME': 'gpt-5-nano', **env_vars}), \
        patch('src.llm.model_manager.completion') as mock_completion:
        
        manager = LLMModelManager()
        manager.chat_completion(messages, **override_params)

        call_args = mock_completion.call_args[1]
        assert call_args['temperature'] == 0.9
        assert call_args['max_tokens'] == 100

