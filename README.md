# MMLU Library

## Description
A Python library for MMLU functionalities.

## Installation
To install the library, clone the repository and install the dependencies:

```bash
git clone https://github.com/VishwamAI/mmlu-experiments.git
cd mmlu-experiments
pip install .
```

## Usage
Here's a basic example of how to use the library:

```python
from mmlu.core import MMLU

# Create an instance of the MMLU class
mmlu = MMLU()

# Example usage of the example_function method
mmlu.example_function()

# Example usage of the read_file method
file_content = mmlu.read_file('path/to/your/file.txt')
print(file_content)

# Example usage of the write_file method
mmlu.write_file('path/to/your/file.txt', 'This is a test content.')

# Example usage of the append_to_file method
mmlu.append_to_file('path/to/your/file.txt', ' This content will be appended.')
```

### HumanAgent Class Usage
The `HumanAgent` class provides advanced functionalities for human-agent interactions, including text generation, sentiment analysis, question answering, summarization, and translation.

```python
from mmlu.human_agent import HumanAgent

# Create an instance of the HumanAgent class
agent = HumanAgent()

# Example usage of the generate_text method
generated_text = agent.generate_text("Once upon a time")
print(generated_text)

# Example usage of the classify_text method
sentiment = agent.classify_text("I love this product!")
print(sentiment)

# Example usage of the answer_question method
answer = agent.answer_question("What is the capital of France?", "France is a country in Europe. The capital of France is Paris.")
print(answer)

# Example usage of the summarize_text method
summary = agent.summarize_text("Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data, without being explicitly programmed.")
print(summary)

# Example usage of the translate_text method
translation = agent.translate_text("Hello, how are you?")
print(translation)
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
