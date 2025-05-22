from transformers import pipeline

# Load model and tokenizer
clf = pipeline("text-classification", model="../models/distilbert-sentiment")

# Predict
text = input("Enter text to analyze: ")
result = clf(text)
print(result)
