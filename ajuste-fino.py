import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE_HF_MODEL = os.environ.get("BASE_HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
CSV_PATH = os.environ.get("CSV_PATH", "data/frutas.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "lora-frutas-adapter")

MAX_LEN = int(os.environ.get("MAX_LEN", "256"))
GPU_MEM = os.environ.get("GPU_MEM", "5GiB")
CPU_MEM = os.environ.get("CPU_MEM", "12GiB")

def build_text(pergunta: str, resposta: str) -> str:
    return f"Pergunta: {pergunta}\nResposta: {resposta}"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Verifique instalação do torch com CUDA.")

    # 1) Dataset
    df = pd.read_csv(CSV_PATH)
    train_ds = Dataset.from_list(
        [{"text": build_text(str(r["pergunta"]), str(r["resposta"]))} for _, r in df.iterrows()]
    )

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_HF_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Quantização 4-bit (compute em FP16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    max_memory = {0: GPU_MEM, "cpu": CPU_MEM}

    # Importante: use dtype= (torch_dtype é deprecated)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_HF_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.float16,
    )
    model.config.use_cache = False

    # 4) LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # 5) Configuração SFT: DESLIGA AMP do Trainer (sem fp16/bf16)
    # Isso evita o GradScaler/AMP cair em bf16 no optimizer.step()
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=10,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        max_length=MAX_LEN,
        gradient_checkpointing=True,

        bf16=False,
        fp16=False,          # <-- chave: remove AMP/GradScaler do Trainer
        max_grad_norm=0.0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_args,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Treino concluído. Adapter salvo em:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
