import openai
import os

class OpenAIIntegration:
    def __init__(self):
        # Retrieve the OpenAI API key from the environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate_text(self, prompt, max_tokens=50):
        """
        Generates text based on the given prompt using OpenAI's GPT-4o model.

        Parameters:
        prompt (str): The input text to generate text from.
        max_tokens (int): The maximum number of tokens to generate.

        Returns:
        str: The generated text.
        """
        response = openai.Completion.create(
            model="gpt-4o",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()

    def analyze_image(self, image_path):
        """
        Analyzes an image using OpenAI's GPT-4o model.

        Parameters:
        image_path (str): The path to the image file to analyze.

        Returns:
        str: The analysis of the image.
        """
        with open(image_path, "rb") as image_file:
            response = openai.Image.create(
                model="gpt-4o",
                image=image_file
            )
        return response.choices[0].text.strip()

    def transcribe_audio(self, audio_path):
        """
        Transcribes audio using OpenAI's Whisper model.

        Parameters:
        audio_path (str): The path to the audio file to transcribe.

        Returns:
        str: The transcription of the audio.
        """
        with open(audio_path, "rb") as audio_file:
            response = openai.Audio.create(
                model="whisper-1",
                audio=audio_file
            )
        return response.choices[0].text.strip()
