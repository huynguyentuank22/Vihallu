# Load trained model and perform inference
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

def load_trained_model(adapter_path):
    """Load trained model from adapter path"""
    
    # Load configuration
    with open(f"{adapter_path}/training_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Load label mappings
    with open(f"{adapter_path}/label_mappings.json", "r", encoding="utf-8") as f:
        label_mappings = json.load(f)
    
    model_id = config["model_id"]
    label2id = {k: int(v) for k, v in label_mappings["label2id"].items()}
    id2label = {int(k): v for k, v in label_mappings["id2label"].items()}
    
    print(f"Loading model: {model_id}")
    print(f"Adapter path: {adapter_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    
    # Load base model with quantization
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        id2label=id2label,
        label2id=label2id
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("✓ Model loaded successfully!")
    print(f"Configuration: {config['task']}")
    print(f"Classes: {config['classes']}")
    
    return model, tokenizer, config, label2id, id2label


def predict(model, tokenizer, context, prompt, response, device="cuda"):
    """Predict hallucination type for given inputs"""
    
    # Create prompt
    text = f"""
Context: {context}
Prompt: {prompt}
Response: {response}
"""
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return predicted_class, probabilities[0].cpu().tolist()


def predict_batch(model, tokenizer, examples, device="cuda"):
    """Predict for multiple examples"""
    predictions = []
    
    for example in examples:
        pred_class, probs = predict(
            model, tokenizer,
            example['context'],
            example['prompt'],
            example['response'],
            device
        )
        predictions.append({
            'predicted_class': pred_class,
            'probabilities': probs,
            'example': example
        })
    
    return predictions


# Example usage
if __name__ == "__main__":
    # Path to trained adapter
    adapter_path = "./adapters/Qwen_Qwen3_1.7B_3class"
    
    # Load model
    model, tokenizer, config, label2id, id2label = load_trained_model(adapter_path)
    
    # Example prediction
    context = "Hà Nội là thủ đô của Việt Nam. Thành phố có diện tích 3.344,7 km²."
    prompt = "Diện tích của Hà Nội là bao nhiêu?"
    response = "Diện tích của Hà Nội là 3.344,7 km²."
    
    pred_class, probs = predict(model, tokenizer, context, prompt, response)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Context: {context}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"\nPredicted Class: {id2label[pred_class]}")
    print(f"Probabilities:")
    for label, prob in zip(['no', 'intrinsic', 'extrinsic'], probs):
        print(f"  {label}: {prob:.4f}")
    print("="*60)
