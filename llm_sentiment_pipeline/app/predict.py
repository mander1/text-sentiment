from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the model and tokenizer
model_path = "../models/distilbert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Input: just one example for now
text = "The food was amazing and the service was great!"

# Tokenize
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()

# Interpret result
label_map = {0: "Negative", 1: "Positive"}
print(f"Text: {text}")
print(f"Prediction: {label_map[prediction]} (confidence: {confidence:.2f})")