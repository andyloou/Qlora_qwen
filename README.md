# QLoRA Fine-Tuning on Qwen2.5-0.5B-Instruct

This project fine-tunes the [Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct) model using QLoRA (Quantized Low-Rank Adapter) technique.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Download Pretrained Model](#1-download-pretrained-model)
  - [2. Load Model with Quantization](#2-load-model-with-quantization)
  - [3. Prepare LoRA Adaptation](#3-prepare-lora-adaptation)
  - [4. Prepare Dataset](#4-prepare-dataset)
  - [5. Fine-tune the Model](#5-fine-tune-the-model)
  - [6. Load Fine-Tuned Model and Inference](#6-load-fine-tuned-model-and-inference)
- [Notes](#notes)

---

## Project Overview

This repository demonstrates how to fine-tune a quantized causal language model using QLoRA techniques. The training is performed on a small custom dataset (`quotes.jsonl`) for fast experimentation.


## Installation

```bash
pip install torch
pip install transformers
pip install datasets
pip install peft
pip install modelscope
```

(Ensure you have CUDA installed for GPU acceleration.)

---

## Usage

### 1. Download Pretrained Model
```python
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')
```

### 2. Load Model with Quantization
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side="right", use_fast=False)
```

### 3. Prepare LoRA Adaptation
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

### 4. Prepare Dataset
```python
from datasets import load_dataset

data = load_dataset('json', data_files="./quotes.jsonl")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

### 5. Fine-tune the Model
```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=50,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/checkpoint-1",
        optim="paged_adamw_8bit",
        save_strategy='steps',
        save_steps=10,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()
trainer.save_model(trainer.args.output_dir)
```

### 6. Load Fine-Tuned Model and Inference
```python
from peft import PeftModel, PeftConfig

peft_model_dir = trainer.args.output_dir
config = PeftConfig.from_pretrained(peft_model_dir)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_dir)

# Example chat
prompt = "\u8be6\u7ec6\u4ecb\u7ecd\u4e00\u4e0b\u5927\u8bed\u8a00\u6a21\u578b,\u8bc4\u4ef7\u4e0b\u4e0e\u6df1\u5ea6\u5b66\u4e60\u7684\u5dee\u5f02"
messages = [
    {"role": "system", "content": "\u4f60\u662f\u4e00\u4e2a\u667a\u80fd\u52a9\u7406."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

gen_kwargs = {"max_length": 512, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**model_inputs, **gen_kwargs)
    outputs = outputs[:, model_inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Notes
- This project uses **QLoRA** (quantized 8-bit/4-bit fine-tuning) for efficient resource usage.
- The model is fine-tuned on a small quote dataset for demonstration purposes.
- LoRA adapters allow fast and memory-efficient updates.

---

Happy Fine-tuning! :rocket:

