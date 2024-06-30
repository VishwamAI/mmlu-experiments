import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Ensure the Hugging Face token is set in the environment
hf_token = os.getenv("Hugging_Face_Hugging_Face")
if not hf_token:
    raise EnvironmentError("Hugging Face token is not set in the environment.")

# Initialize the pipeline for sentiment analysis with the token
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=hf_token)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = sentiment_pipeline("Hello, I love using Hugging Face!")
    print("Token is valid. Sentiment analysis result:", result)
except Exception as e:
    print("Error during Hugging Face API call:", str(e))
