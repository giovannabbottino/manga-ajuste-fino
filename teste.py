import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_HF_MODEL = os.environ.get("BASE_HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "lora-frutas-adapter")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível.")

    # 1) Quantização 4-bit (bnb) — sem device_map/dispatch
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_HF_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Modelo base quantizado -> CUDA direto (evita accelerate dispatch)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_HF_MODEL,
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
    )
    base.to("cuda")
    base.eval()

    # 4) Acopla o adaptador LoRA
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.to("cuda")
    model.eval()

    # 5) Pergunta
    question = "O que é manga?"
    messages = [{"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # 6) Geração (conservadora pra caber em 6GB)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Reduz fragmentação (ajuda em GPUs pequenas)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
