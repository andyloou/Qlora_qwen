from datetime import datetime
now = datetime.now()
time_str = now.strftime('%Y-%m-%d %H:%M:%S')
print(time_str)

from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig


device = "auto" 


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map=device,trust_remote_code=True,torch_dtype=torch.float16,quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True,padding_side="right",use_fast=False)
model.gradient_checkpointing_enable

print(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print(model)
print_trainable_parameters(model)


# Verifying the datatypes.
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items():
    total += v
for k, v in dtypes.items():
    print(k, v, v / total)

"""### Training"""

data = load_dataset('json',data_files="./quotes.jsonl")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
print(data)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=50,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/checkpoint-1"+time_str,
        optim="paged_adamw_8bit",
        save_strategy = 'steps',
        save_steps = 10,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model(trainer.args.output_dir)


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_dir = trainer.args.output_dir
config = PeftConfig.from_pretrained(peft_model_dir)
print(config)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True,  device_map=device,
    torch_dtype=torch.float16, quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_dir)

print(model)
# 模拟对话
prompt = "详细介绍一下大语言模型,评价下与深度学习的差异"
messages = [
    {"role": "system", "content": "你是一个智能助理."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

gen_kwargs = {"max_length": 512, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**model_inputs, **gen_kwargs)
    outputs = outputs[:, model_inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))