import os
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    model_class: str = "AutoModelForCausalLM"
    proc_class: str = "AutoTokenizer"
    is_vlm: bool = False
    slow_tokenizer: bool = False
    batch_size_divide: int = 1
    num_params: float | None = None


REGISTRY = {
    "gemma3": ModelSpec("google/gemma-3-27b-it", "Gemma3ForConditionalGeneration", "AutoProcessor",
                        is_vlm=True, num_params=27.0),
    "llama-3.3-70b": ModelSpec("meta-llama/Llama-3.3-70B-Instruct", batch_size_divide=2, num_params=70.0),
    "qwen2.5-72b": ModelSpec("Qwen/Qwen2.5-72B-Instruct", batch_size_divide=2, num_params=72.0),
    # The fast Mistral tokenizer mis-splits some batched inputs, so the slow one is used.
    "mistral-24b": ModelSpec("mistralai/Mistral-Small-24B-Instruct-2501", slow_tokenizer=True, num_params=24.0),
    "qwen2.5-05b": ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", num_params=0.5),
    "qwen2.5-1b": ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", num_params=1.5),
    "qwen2.5-3b": ModelSpec("Qwen/Qwen2.5-3B-Instruct", num_params=3.0),
    "qwen2.5-7b": ModelSpec("Qwen/Qwen2.5-7B-Instruct", num_params=7.0),
    "qwen2.5-14b": ModelSpec("Qwen/Qwen2.5-14B-Instruct", num_params=14.0),
    "qwen2.5-32b": ModelSpec("Qwen/Qwen2.5-32B-Instruct", batch_size_divide=2, num_params=32.0),
}

# Judges scored through OpenRouter (score_openrouter.py), by OpenRouter slug.
CLOSED_MODELS = ["openai/gpt-5.4-nano"]


def tokenizer_of(proc, spec: ModelSpec):
    return proc.tokenizer if spec.is_vlm else proc


def load_model(spec: ModelSpec):
    import torch
    import transformers

    token = os.getenv("HF_TOKEN")
    proc = None
    if spec.slow_tokenizer:
        try:
            proc = transformers.AutoTokenizer.from_pretrained(spec.name, token=token, use_fast=False)
        except Exception as e:
            print(f"Could not load the slow tokenizer of {spec.name}, using the fast one: {e}")
    if proc is None:
        proc = getattr(transformers, spec.proc_class).from_pretrained(spec.name, token=token)

    model = getattr(transformers, spec.model_class).from_pretrained(
        spec.name, token=token, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    tok = tokenizer_of(proc, spec)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, proc
