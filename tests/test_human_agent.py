import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmlu.human_agent import HumanAgent

# Set the Hugging Face token globally for all tests
@pytest.fixture(scope="module", autouse=True)
def set_hugging_face_token():
    os.environ["Hugging_Face_Hugging_Face"] = os.getenv("Hugging_Face_Hugging_Face")

@pytest.fixture
def human_agent():
    return HumanAgent()

@patch('mmlu.human_agent.pipeline')
def test_generate_text(mock_pipeline, human_agent):
    mock_model = MagicMock()
    mock_model.return_value = [{"generated_text": "Once upon a time, there was a magical land."}]
    mock_pipeline.return_value = mock_model

    prompt = "Once upon a time"
    result = human_agent.generate_text(prompt)
    assert isinstance(result, str)
    assert len(result) > len(prompt)

@patch('mmlu.human_agent.pipeline')
def test_classify_text(mock_pipeline, human_agent):
    mock_model = MagicMock()
    mock_model.return_value = [{"label": "POSITIVE", "score": 0.99}]
    mock_pipeline.return_value = mock_model

    text = "I love this product!"
    result = human_agent.classify_text(text)
    assert isinstance(result, dict)
    assert "label" in result
    assert "score" in result

@patch('mmlu.human_agent.pipeline')
def test_answer_question(mock_pipeline, human_agent):
    mock_model = MagicMock()
    mock_model.return_value = {"answer": "Paris"}
    mock_pipeline.return_value = mock_model

    question = "What is the capital of France?"
    context = "France is a country in Europe. The capital of France is Paris."
    result = human_agent.answer_question(question, context)
    assert isinstance(result, str)
    assert result.lower() == "paris"

@patch('mmlu.human_agent.pipeline')
def test_summarize_text(mock_pipeline, human_agent):
    mock_model = MagicMock()
    mock_model.return_value = [{"summary_text": "Machine learning is a field of AI that uses statistical techniques to learn from data."}]
    mock_pipeline.return_value = mock_model

    text = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data, without being explicitly programmed."
    result = human_agent.summarize_text(text)
    assert isinstance(result, str)
    assert len(result) < len(text)

@patch('mmlu.human_agent.pipeline')
def test_translate_text(mock_pipeline, human_agent):
    mock_model = MagicMock()
    mock_model.return_value = [{"translation_text": "Bonjour, comment ça va?"}]
    mock_pipeline.return_value = mock_model

    text = "Hello, how are you?"
    result = human_agent.translate_text(text)
    assert isinstance(result, str)
    assert "Bonjour" in result
