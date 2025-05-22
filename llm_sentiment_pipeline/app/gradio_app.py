import gradio as gr
from transformers import pipeline

# Load model and tokenizer
clf = pipeline("text-classification", model="models/distilbert-sentiment")

# Define prediction function
def classify_sentiment(text):
    result = clf(text)[0]
    label = result["label"]
    score = round(result["score"], 3)
    return f"Sentiment: {label} (Confidence: {score})"

# Build Gradio interface
iface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a review or comment here..."),
    outputs="text",
    title="LLM Sentiment Classifier",
    description="Enter a sentence and get its predicted sentiment (positive or negative)."
)

if __name__ == "__main__":
    iface.launch()
