import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download

# Tải mô hình gốc từ ModelScope
print(" Đang tải mô hình gốc từ ModelScope...")
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')

# Đường dẫn thư mục chứa trọng số LoRA (sau khi huấn luyện)
peft_model_dir = "outputs/checkpoint-50"  

#  Tải mô hình gốc
print(" Đang tải mô hình gốc...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

#  Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

#  Nạp trọng số LoRA vào mô hình gốc
print(" Đang tải trọng số LoRA từ", peft_model_dir)
model = PeftModel.from_pretrained(model, peft_model_dir)

print("Mô hình đã sẵn sàng để sử dụng!")

# Hàm chạy thử mô hình (Inference)
def chat_with_model(prompt):
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Cấu hình sinh văn bản
    gen_kwargs = {"max_length": 512, "do_sample": True, "top_k": 1}

    # Chạy mô hình
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]  # Cắt bỏ phần đầu
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Chạy thử mô hình
while True:
    user_input = input(" Nhập câu hỏi (hoặc gõ 'exit' để thoát): ")
    if user_input.lower() == "exit":
        break
    answer = chat_with_model(user_input)
    print(f"Trợ lý AI: {answer}\n")
