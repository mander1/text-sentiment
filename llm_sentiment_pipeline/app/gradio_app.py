import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model and tokenizer from local directory
model_path = "../models/distilbert-sentiment"  # Adjust path if needed
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create pipeline using local model/tokenizer
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define prediction function
def classify_sentiment(text):
    results = clf(text, top_k=2)
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive"
    }

    output_lines = []
    for result in results:
        label = label_map.get(result["label"], result["label"])
        score = round(result["score"] * 100, 1)
        output_lines.append(f"{label}: {score}%")

    return "\n".join(output_lines)



# Build Gradio interface
iface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a review or comment here..."),
    outputs="text",
    title="LLM Sentiment Classifier",
    description="Enter a sentence and get its predicted sentiment (positive or negative)."
)

if __name__ == "__main__":
    iface.launch()
