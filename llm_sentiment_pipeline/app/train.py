from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import transformers
print(transformers.__version__)
print(transformers.__file__)


# Load and prepare dataset
dataset = load_dataset("yelp_polarity")
small_train = dataset["train"].shuffle(seed=42).select(range(500))
small_test = dataset["test"].shuffle(seed=42).select(range(100))

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = small_train.map(tokenize_fn, batched=True)
test_dataset = small_test.map(tokenize_fn, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="../models/distilbert-sentiment",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_total_limit=1,
    logging_dir="../logs",  # Optional: for TensorBoard
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train and save
trainer.train()
trainer.save_model("../models/distilbert-sentiment")
tokenizer.save_pretrained("../models/distilbert-sentiment")
