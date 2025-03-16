import torch, transformers

# Check if GPU is available
print("Transformers version:", transformers.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories")
dataset = dataset["train"].select(range(1000))
dataset = dataset.train_test_split(test_size=0.1, seed=1)

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model_config = GPT2Config(
    n_positions=512,
    n_embd=384, n_layer=6, n_head=6,
    vocab_size=tokenizer.vocab_size,
    activation_function="gelu_new",
)
model = GPT2LMHeadModel(model_config).to(device)
model.loss_type = "ForCausalLM"

# Tokenize the dataset
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    output = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": tokens["input_ids"],
    }
    return output
tokenized_datasets = dataset.map(tokenize_function, remove_columns=["text"], batched=True)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    num_train_epochs=20,
    metric_for_best_model="eval_loss",
    warmup_steps=500, logging_steps=1,
    fp16=True if torch.cuda.is_available() else False,
    disable_tqdm=True, report_to=None,
)

from transformers import TrainerCallback
class GenerationCallback(TrainerCallback):
    def __init__(self):
        self.eval_loss = float('inf')
        self.train_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_loss = min(self.train_loss, logs['loss'])
        if 'eval_loss' in logs:
            self.eval_loss = min(self.eval_loss, logs['eval_loss'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.eval_loss = min(self.eval_loss, metrics['eval_loss'])

            epoch = int(state.epoch)
            total_epochs = int(args.num_train_epochs)

            prompts = "Lily was a little girl"
            tokens = tokenizer(prompts, return_tensors="pt")
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            output = model.generate(
                input_ids, attention_mask=attention_mask,
                max_length=20, temperature=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            text = tokenizer.decode(output[0], skip_special_tokens=False)
            text = text.replace("\n", " ")

            message = f"Epoch {epoch:4d} / {total_epochs:4d}, "
            message += f"Training Loss: {self.train_loss:.4f}, "
            message += f"Eval Loss: {self.eval_loss:.4f}, "
            message += f"Sample: {text}"
            print(message)

# Initialize Trainer
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[GenerationCallback()],
)

from transformers import PrinterCallback
trainer.remove_callback(PrinterCallback)
trainer.train()
