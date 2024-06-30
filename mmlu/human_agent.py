from transformers import pipeline
import os

class HumanAgent:
    def __init__(self):
        # Retrieve the Hugging Face token from the environment variable
        hf_token = os.getenv("Hugging_Face_Hugging_Face")

        # Initialize the pipelines for various tasks with the token
        self.text_generator = pipeline("text-generation", model="gpt-2", use_auth_token=hf_token)
        self.text_classifier = pipeline("sentiment-analysis", use_auth_token=hf_token)
        self.question_answerer = pipeline("question-answering", use_auth_token=hf_token)
        self.summarizer = pipeline("summarization", use_auth_token=hf_token)
        self.translator = pipeline("translation_en_to_fr", use_auth_token=hf_token)

    def generate_text(self, prompt, max_length=50):
        """
        Generates text based on the given prompt.

        Parameters:
        prompt (str): The input text to generate text from.
        max_length (int): The maximum length of the generated text.

        Returns:
        str: The generated text.
        """
        return self.text_generator(prompt, max_length=max_length)[0]['generated_text']

    def classify_text(self, text):
        """
        Classifies the sentiment of the given text.

        Parameters:
        text (str): The input text to classify.

        Returns:
        str: The sentiment classification.
        """
        return self.text_classifier(text)[0]

    def answer_question(self, question, context):
        """
        Answers a question based on the given context.

        Parameters:
        question (str): The question to answer.
        context (str): The context to find the answer in.

        Returns:
        str: The answer to the question.
        """
        return self.question_answerer(question=question, context=context)['answer']

    def summarize_text(self, text):
        """
        Summarizes the given text.

        Parameters:
        text (str): The input text to summarize.

        Returns:
        str: The summary of the text.
        """
        return self.summarizer(text)[0]['summary_text']

    def translate_text(self, text):
        """
        Translates the given text from English to French.

        Parameters:
        text (str): The input text to translate.

        Returns:
        str: The translated text.
        """
        return self.translator(text)[0]['translation_text']
