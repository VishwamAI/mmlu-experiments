import pytest
import os
import sys

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

def test_generate_text(human_agent):
    prompt = "Once upon a time"
    result = human_agent.generate_text(prompt)
    assert isinstance(result, str)
    assert len(result) > len(prompt)

def test_classify_text(human_agent):
    text = "I love this product!"
    result = human_agent.classify_text(text)
    assert isinstance(result, dict)
    assert "label" in result
    assert "score" in result

def test_answer_question(human_agent):
    question = "What is the capital of France?"
    context = "France is a country in Europe. The capital of France is Paris."
    result = human_agent.answer_question(question, context)
    assert isinstance(result, str)
    assert result.lower() == "paris"

def test_summarize_text(human_agent):
    text = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data, without being explicitly programmed."
    result = human_agent.summarize_text(text)
    assert isinstance(result, str)
    assert len(result) < len(text)

def test_translate_text(human_agent):
    text = "Hello, how are you?"
    result = human_agent.translate_text(text)
    assert isinstance(result, str)
    assert "Bonjour" in result
