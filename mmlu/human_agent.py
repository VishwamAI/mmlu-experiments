from transformers import pipeline
import os

class HumanAgent:
    def __init__(self):
        # Retrieve the Hugging Face token from the environment variable
        self.hf_token = os.getenv("Hugging_Face_Hugging_Face")
        self._text_generator = None
        self._text_classifier = None
        self._question_answerer = None
        self._summarizer = None
        self._translator = None
        self._speech_recognizer = None

    @property
    def text_generator(self):
        if self._text_generator is None:
            self._text_generator = pipeline("text-generation", model="distilgpt2", use_auth_token=self.hf_token)
        return self._text_generator

    @property
    def text_classifier(self):
        if self._text_classifier is None:
            self._text_classifier = pipeline("sentiment-analysis", use_auth_token=self.hf_token)
        return self._text_classifier

    @property
    def question_answerer(self):
        if self._question_answerer is None:
            self._question_answerer = pipeline("question-answering", use_auth_token=self.hf_token)
        return self._question_answerer

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = pipeline("summarization", use_auth_token=self.hf_token)
        return self._summarizer

    @property
    def translator(self):
        if self._translator is None:
            self._translator = pipeline("translation_en_to_fr", use_auth_token=self.hf_token)
        return self._translator

    @property
    def speech_recognizer(self):
        if self._speech_recognizer is None:
            self._speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", use_auth_token=self.hf_token)
        return self._speech_recognizer

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

    def transcribe_audio(self, audio_path):
        """
        Transcribes audio using the speech recognition pipeline.

        Parameters:
        audio_path (str): The path to the audio file to transcribe.

        Returns:
        str: The transcription of the audio.
        """
        return self.speech_recognizer(audio_path)[0]['text']
