from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from transformers import Trainer

# Load model and tokenizer
model_path = r"F:\Becoming a software person\machine learning\text-sentiment\llm_sentiment_pipeline\models\distilbert-sentiment"

print("Exists:", os.path.exists(model_path))
print("Files:", os.listdir(model_path))

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Load test dataset (adjust size as needed)
dataset = load_dataset("yelp_polarity")
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

# Tokenize test data
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

test_dataset = test_dataset.map(tokenize_fn, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create Trainer instance
trainer = Trainer(model=model)

# Run predictions
print("Running evaluation...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Accuracy and classification report
print("\nAccuracy:", accuracy_score(labels, preds))
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

# Confusion matrix
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
