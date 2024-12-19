import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Set the device to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device) # using the finbert model
labels = ["positive", "negative", "neutral"] # sentiment labels

# Function to estimate the sentiment of a news text
def estimate_sentiment(news_text):
    if not news_text:
        return 0, "neutral"
    
    filtered_text = [txt for txt in news_text if txt.strip()]
    if not filtered_text:
        logging.info("All provided texts are empty or whitespace")
        return 0.0, "neutral"
    try:
        inputs = tokenizer(filtered_text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            res = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]

        result = torch.nn.functional.softmax(torch.sum(res, 0), dim=-1)
        return result[torch.argmax(result)], labels[torch.argmax(result)] # return the probability and sentiment
    
    except Exception as e:
        logging.error(f"Error during sentiment estimation: {e}")
        return 0.0, "neutral"