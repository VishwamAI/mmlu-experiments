import os
import sys
import pytest
from unittest.mock import patch
from mmlu.integrations.openai_integration import OpenAIIntegration

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def openai_integration():
    return OpenAIIntegration()

@patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
@patch('openai.Completion.create')
def test_generate_text(mock_create, openai_integration):
    mock_create.return_value = type('obj', (object,), {'choices': [type('obj', (object,), {'text': 'Paris'})]})
    prompt = "What is the capital of France?"
    result = openai_integration.generate_text(prompt)
    assert isinstance(result, str)
    assert len(result) > 0

@patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
@patch('openai.Image.create')
def test_analyze_image(mock_create, openai_integration):
    mock_create.return_value = type('obj', (object,), {'choices': [type('obj', (object,), {'text': 'Image analysis result'})]})
    image_path = "tests/sample_image.png"
    result = openai_integration.analyze_image(image_path)
    assert isinstance(result, str)
    assert len(result) > 0

@patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
@patch('openai.Audio.create')
def test_transcribe_audio(mock_create, openai_integration):
    mock_create.return_value = type('obj', (object,), {'choices': [type('obj', (object,), {'text': 'Audio transcription result'})]})
    audio_path = "tests/sample_audio.wav"
    result = openai_integration.transcribe_audio(audio_path)
    assert isinstance(result, str)
    assert len(result) > 0
