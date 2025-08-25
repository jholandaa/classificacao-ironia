import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Caminhos
train_path = "data/train.jsonl"
test_path = "data/test.csv"
output_path = "output/submission.csv"

# Carregar datasets
train_data = pd.read_json(train_path, lines=True)
test_data = pd.read_csv(test_path)

# Preparar dataset HuggingFace
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = Dataset.from_pandas(train_data)
train_dataset = train_dataset.map(tokenize, batched=True)

# Modelo pré-treinado
model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)

# Configuração de treino
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Treinar
trainer.train()

# Previsões no conjunto de teste
test_dataset = Dataset.from_pandas(test_data)
test_dataset = test_dataset.map(tokenize, batched=True)

preds = trainer.predict(test_dataset)
labels = torch.argmax(torch.tensor(preds.predictions), dim=1)

# Gerar submissão
submission = pd.DataFrame({
    "id": test_data["id"],
    "label": labels.numpy()
})
submission.to_csv(output_path, index=False)
print(f"Submissão salva em {output_path}")
