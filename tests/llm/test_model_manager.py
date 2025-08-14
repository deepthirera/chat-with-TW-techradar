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

def test_model_default_from_config(env_vars):
    """Test that model defaults to config value when not in env"""
    with patch.dict(os.environ, {'model_name': '', 'embedding_model_name': '', **env_vars}, clear=True), \
        patch('src.llm.model_manager.completion', return_value={'choices': [{'message': {'content': 'test-response'}}]}):
        manager = LLMModelManager()
        manager.chat_completion([])
        assert manager.model_alias == 'ollama-mistral'
        assert manager.embedding_model_alias == 'ollama-nominic'

@pytest.mark.parametrize("model_alias, expected_params", [
    ('openai-gpt4', {
        'model_alias': 'openai-gpt4',
        'model': 'gpt-4-1106-preview',
        'max_tokens': 4096,
        'temperature': 0.7
    }),
    ('ollama-mistral', {
        'model_alias': 'ollama-mistral',
        'model': "ollama/mistral",
        'api_base': 'http://localhost:11434',
        'max_tokens': 4096,
        'temperature': 0.7
    })
])
def test_chat_completion_parameters(env_vars, model_alias, expected_params):
    """Test that chat completion is called with correct parameters"""
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch.dict(os.environ, {'model_name': model_alias, **env_vars}), \
        patch('src.llm.model_manager.completion') as mock_completion:
        
        manager = LLMModelManager()
        manager.chat_completion(messages)
        assert manager.model_alias == model_alias
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        assert call_args['model'] == expected_params['model']
        assert call_args['messages'] == messages
        for key, value in expected_params.items():
            assert call_args[key] == value

def test_override_parameters_in_completion(env_vars):
    """Test that passed kwargs override default parameters in completion"""
    messages = [{"role": "user", "content": "Hello"}]
    override_params = {
        'temperature': 0.9,
        'max_tokens': 100
    }
    
    with patch.dict(os.environ, {'model_name': 'openai-gpt35', **env_vars}), \
        patch('src.llm.model_manager.completion') as mock_completion:
        
        manager = LLMModelManager()
        manager.chat_completion(messages, **override_params)

        call_args = mock_completion.call_args[1]
        assert call_args['temperature'] == 0.9
        assert call_args['max_tokens'] == 100
