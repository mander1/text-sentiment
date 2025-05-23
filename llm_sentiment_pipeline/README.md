# LLM Sentiment Classifier

A simple machine learning pipeline that uses a fine-tuned DistilBERT model to classify sentiment in text as **Positive** or **Negative**. This project demonstrates training a transformer model using Hugging Face `transformers` and `datasets`, and serving predictions via a Gradio web interface.

---

## Features

- Fine-tunes `distilbert-base-uncased` on the Yelp Polarity dataset.
- Classifies text sentiment as Positive or Negative.
- Simple Gradio web app for easy interaction.
- Local-only solution — no Hugging Face API key or cloud deployment needed.

---

## Project Structure

<pre> ```text llm_sentiment_pipeline/ │ ├── app/ │ ├── train.py # Fine-tunes the sentiment classification model │ ├── gradio_app.py # Gradio interface for sentiment prediction │ ├── predict.py # (Optional) Script for command-line inference │ └── models/ │ └── distilbert-sentiment/ # Saved model and tokenizer │ ├── README.md └── environment.yml # (Optional) Conda environment file ``` </pre>

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/llm_sentiment_pipeline.git
cd llm_sentiment_pipeline/app
```

## Conda

conda create -n bertie python=3.10
conda activate bertie

## Dependencies

pip install -r requirements.txt

## Training

Fine-tune the DistilBERT model on a subset of the Yelp Polarity dataset:

python train.py

## Launch the local web interface:

python gradio_app.py
open your browser to: http://127.0.0.1:7860

## Sample Usage

Input: 'I absolutely loved the experience and would recommend it to everyone!'

Output: Positive: 94.6% / Negative: 5.4%
