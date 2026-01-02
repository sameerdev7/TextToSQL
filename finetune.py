import os
import torch
import argparse
from datasets import load_dataset
from typing import Optional
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

TRAINER_OUTPUT_DIR = "trainer_output"

PHI_SYSTEM_PROMPT = (
    "You are an expert Text-to-SQL assistant. "
    "Return ONLY executable SQL for the given question and schema. "
    "Do not include explanations, comments, or markdown. "
    "Prefer ANSI SQL; use tables/columns exactly as provided."
)

PHI_CHAT_FALLBACK_TEMPLATE = (
    "<|system|>\n{system}\n"
    "<|user|>\n{user}\n"
    "<|assistant|>\n{assistant}"
)

def _coalesce(s: Optional[str]) -> str:
    return "" if s is None else str(s)

def get_dataset(
    data_file: str,
    eos_token: str,
    dataset_size: Optional[int] = None,
    tokenizer=None,
    system_prompt: str = PHI_SYSTEM_PROMPT,
):
    ds = load_dataset("json", data_files={"train": data_file}, split="train")

    if dataset_size is not None:
        ds = ds.select(range(min(dataset_size, len(ds))))

    def build_example(inst: str, inp: str, out: str) -> str:
        inst = _coalesce(inst).strip()
        inp = _coalesce(inp).strip()
        out = _coalesce(out).strip()

        user_msg = inst if not inp else f"{inst}\n\n{inp}"

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": out},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ) + eos_token

        return PHI_CHAT_FALLBACK_TEMPLATE.format(
            system=system_prompt,
            user=user_msg,
            assistant=out,
        ) + eos_token

    def preprocess(batch):
        return {
            "text": [
                build_example(i, inp, o)
                for i, inp, o in zip(
                    batch["instruction"], batch["input"], batch["output"]
                )
            ]
        }

    return ds.map(preprocess, remove_columns=ds.column_names, batched=True)

def main(args):
    print(f"Loading model: {args.model_name}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype_map[args.dtype],
        device_map="auto",
    )

    # REQUIRED FOR TRAINING
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.seq_length

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            target_modules=["qkv_proj", "o_proj", "fc1", "fc2"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )

    # REQUIRED FOR PHI
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print(
        f"Trainable parameters = "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    print(f"Loading dataset with {args.dataset_size} samples...")
    dataset = get_dataset(
        args.data_file,
        tokenizer.eos_token,
        args.dataset_size,
        tokenizer,
        PHI_SYSTEM_PROMPT,
    )

    os.makedirs(TRAINER_OUTPUT_DIR, exist_ok=True)

    config = {
        "per_device_train_batch_size": args.batch_size,
        "num_train_epochs": args.num_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "output_dir": TRAINER_OUTPUT_DIR,
        "remove_unused_columns": False,
        "seed": 42,
        "dataset_text_field": "text",
        "packing": False,
        "report_to": "tensorboard",
        "logging_dir": args.log_dir,
        "logging_steps": args.logging_steps,
    }

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**config),
    )

    trainer_stats = trainer.train()

    trainer.model.save_pretrained(TRAINER_OUTPUT_DIR)
    tokenizer.save_pretrained(TRAINER_OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    for k, v in trainer_stats.metrics.items():
        print(f"{k}: {v}")
    print("=" * 60)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", type=str, default="training_data.jsonl")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-3.5-mini-instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=500)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="logs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

