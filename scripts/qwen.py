# Fine-tuning PhoBERT/Qwen for Vietnamese Hallucination Detection (3-class)
# !pip install transformers datasets peft accelerate bitsandbytes scikit-learn evaluate -q

## Step 1: Import Libraries and Setup
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import json
import gc

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Memory management helper
def clear_memory():
    """Clear GPU cache to free memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")

clear_memory()

## Step 2: Data Preprocessing
def clean_text(text: str) -> str:
    import re, unicodedata
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"[(){}\[\]]", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s\.,;:!?\"'%\-…]", " ", text)
    text = re.sub(r"\.{2,}", "…", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load data
train_df = pd.read_csv("/kaggle/input/vihallu/data/vihallu_train.csv")
dev_df = pd.read_csv("/kaggle/input/vihallu/data/vihallu_dev.csv")
test_df = pd.read_csv("/kaggle/input/vihallu/data/vihallu_test.csv")

# Clean text
for df in [train_df, dev_df, test_df]:
    df['context'] = df['context'].apply(clean_text)
    df['prompt'] = df['prompt'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)

print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
print(f"Label distribution:\n{train_df['label'].value_counts()}")

# Create label mappings
label2id = {"no": 0, "intrinsic": 1, "extrinsic": 2}
id2label = {0: "no", 1: "intrinsic", 2: "extrinsic"}

print(f"\nTraining with all 3 classes:")
print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
print(train_df['label'].value_counts())

## Step 3: Load Base Model and Tokenizer
model_id = "Qwen/Qwen2.5-0.5B"

# Create model name from model_id for output directory
model_name = model_id.replace("/", "_").replace("-", "_")
print(f"Model name for output: {model_name}")

# Quantization config for efficient training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Tokenizer loaded: {model_id}")

## Step 4: Tokenization Function
def create_prompt(context, prompt, response):
    """Create instruction prompt for classification"""
    return f"""
Context: {context}
Prompt: {prompt}
Response: {response}
"""

def tokenize_function(examples, max_length=512):
    """Tokenize examples - works with HF Dataset batches"""
    # Create prompts from batch
    prompts = [
        create_prompt(context, prompt, response)
        for context, prompt, response in zip(
            examples['context'],
            examples['prompt'],
            examples['response']
        )
    ]

    # Tokenize (don't use return_tensors, let Trainer handle it)
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,  # Let DataCollator handle padding dynamically
        max_length=max_length
    )

    # Convert labels to integers
    tokenized["labels"] = [label2id[label] for label in examples['label']]
    return tokenized

## Step 5: Prepare Datasets
def prepare_dataset(train_df, dev_df):
    """Convert pandas df to HF dataset with tokenization"""
    # Convert to HF Dataset first
    train_dataset = Dataset.from_pandas(train_df[['context', 'prompt', 'response', 'label']])
    dev_dataset = Dataset.from_pandas(dev_df[['context', 'prompt', 'response', 'label']])
    
    # Then tokenize using .map() for proper batching
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['context', 'prompt', 'response', 'label'],
        desc="Tokenizing train dataset"
    )
    
    dev_dataset = dev_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['context', 'prompt', 'response', 'label'],
        desc="Tokenizing dev dataset"
    )

    return DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset
    })

# Prepare full dataset
print("Preparing dataset with all 3 classes...")
dataset = prepare_dataset(train_df, dev_df)
print("\nDataset ready!")
print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")

## Step 6: Define LoRA Configuration
def get_lora_config():
    """LoRA configuration for efficient fine-tuning"""
    return LoraConfig(
        r=16,                          # LoRA rank
        lora_alpha=32,                 # LoRA scaling
        target_modules=[               # Qwen3 attention modules
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

## Step 7: Define Metrics
def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    # Print detailed metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['no', 'intrinsic', 'extrinsic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))
    print("="*60 + "\n")

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

## Step 8: Train Model with All 3 Classes
print("=" * 60)
print("TRAINING: 3-Class Hallucination Detection")
print("Classes: no, intrinsic, extrinsic")
print("=" * 60)

# Clear memory before loading model
clear_memory()

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,  # no, intrinsic, extrinsic
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    id2label=id2label,
    label2id=label2id
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Apply LoRA
lora_config = get_lora_config()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Create output directories
results_dir = f"./results/{model_name}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory created: {results_dir}")

# Training arguments (SPEED OPTIMIZED)
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=2,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    bf16=True,
    # gradient_checkpointing=True,
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
    optim="adamw_torch_fused"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train
print("\nStarting training...")
train_result = trainer.train()

# Save adapter
output_dir = f"./adapters/{model_name}"
os.makedirs(output_dir, exist_ok=True)
print(f"Adapter directory created: {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✓ Model trained and saved to {output_dir}")

# Save label mappings
with open(f"{output_dir}/label_mappings.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

# Save training configuration and hyperparameters
training_config = {
    "model_id": model_id,
    "model_name": model_name,
    "task": "3-class hallucination detection",
    "classes": ["no", "intrinsic", "extrinsic"],
    "num_labels": 3,
    "dataset_sizes": {
        "train": len(dataset["train"]),
        "validation": len(dataset["validation"])
    },
    "hyperparameters": {
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "max_length": 512,
        "bf16": training_args.bf16,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "early_stopping_patience": 3
    },
    "lora_config": {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": list(lora_config.target_modules) if isinstance(lora_config.target_modules, set) else lora_config.target_modules
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16"
    },
    "best_validation_metrics": {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric
    },
    "training_time_seconds": train_result.metrics.get("train_runtime", 0)
}

with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

print(f"✓ Training config saved to {output_dir}/training_config.json")

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)

## Step 9: Evaluate on Test Set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

# Prepare test dataset
test_dataset = Dataset.from_pandas(test_df[['context', 'prompt', 'response', 'label']])
test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['context', 'prompt', 'response', 'label'],
    desc="Tokenizing test dataset"
)

# Predict
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids
pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# Calculate metrics
test_accuracy = accuracy_score(true_labels, pred_labels)
test_f1_macro = f1_score(true_labels, pred_labels, average='macro')
test_f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

print(f"\nTest Set Results:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Macro: {test_f1_macro:.4f}")
print(f"F1 Weighted: {test_f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic']))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, pred_labels)
print(cm)

# Save test results
results = {
    "model_info": {
        "model_id": model_id,
        "model_name": model_name,
        "task": "3-class hallucination detection"
    },
    "test_metrics": {
        "accuracy": float(test_accuracy),
        "f1_macro": float(test_f1_macro),
        "f1_weighted": float(test_f1_weighted)
    },
    "test_size": len(test_df),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(true_labels, pred_labels, 
                                                   target_names=['no', 'intrinsic', 'extrinsic'],
                                                   output_dict=True),
    "per_class_metrics": {
        "no": {
            "precision": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['no']['precision'],
            "recall": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['no']['recall'],
            "f1-score": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['no']['f1-score'],
            "support": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['no']['support']
        },
        "intrinsic": {
            "precision": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['intrinsic']['precision'],
            "recall": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['intrinsic']['recall'],
            "f1-score": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['intrinsic']['f1-score'],
            "support": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['intrinsic']['support']
        },
        "extrinsic": {
            "precision": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['extrinsic']['precision'],
            "recall": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['extrinsic']['recall'],
            "f1-score": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['extrinsic']['f1-score'],
            "support": classification_report(true_labels, pred_labels, target_names=['no', 'intrinsic', 'extrinsic'], output_dict=True)['extrinsic']['support']
        }
    }
}

with open(f"{output_dir}/test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✓ Test results saved to {output_dir}/test_results.json")

# Save detailed predictions for each test example
test_predictions = []
for i in range(len(test_df)):
    test_predictions.append({
        "id": i,
        "context": test_df.iloc[i]['context'],
        "prompt": test_df.iloc[i]['prompt'],
        "response": test_df.iloc[i]['response'],
        "true_label": id2label[true_labels[i]],
        "predicted_label": id2label[pred_labels[i]],
        "is_correct": bool(true_labels[i] == pred_labels[i]),
        "probabilities": {
            "no": float(pred_probs[i][0]),
            "intrinsic": float(pred_probs[i][1]),
            "extrinsic": float(pred_probs[i][2])
        },
        "confidence": float(pred_probs[i][pred_labels[i]])
    })

# Save all predictions
with open(f"{output_dir}/test_predictions.json", "w", encoding="utf-8") as f:
    json.dump(test_predictions, f, ensure_ascii=False, indent=2)

print(f"✓ Detailed predictions saved to {output_dir}/test_predictions.json")

# Save only misclassified examples for error analysis
misclassified = [pred for pred in test_predictions if not pred['is_correct']]
with open(f"{output_dir}/test_misclassified.json", "w", encoding="utf-8") as f:
    json.dump(misclassified, f, ensure_ascii=False, indent=2)

print(f"✓ Misclassified examples saved to {output_dir}/test_misclassified.json ({len(misclassified)} examples)")

# Create predictions DataFrame for easy analysis
predictions_df = pd.DataFrame({
    'id': range(len(test_df)),
    'context': test_df['context'].values,
    'prompt': test_df['prompt'].values,
    'response': test_df['response'].values,
    'true_label': [id2label[label] for label in true_labels],
    'predicted_label': [id2label[label] for label in pred_labels],
    'is_correct': true_labels == pred_labels,
    'prob_no': pred_probs[:, 0],
    'prob_intrinsic': pred_probs[:, 1],
    'prob_extrinsic': pred_probs[:, 2],
    'confidence': [pred_probs[i][pred_labels[i]] for i in range(len(pred_labels))]
})

predictions_df.to_csv(f"{output_dir}/test_predictions.csv", index=False, encoding='utf-8-sig')
print(f"✓ Predictions CSV saved to {output_dir}/test_predictions.csv")

print("="*60)

# Free memory
clear_memory()
