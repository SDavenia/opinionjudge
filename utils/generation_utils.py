import torch
from tqdm import tqdm

from utils.models_utils import tokenizer_of


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def chat_text(tok, spec, prompt):
    content = [{"type": "text", "text": prompt}] if spec.is_vlm else prompt
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def tokenize(proc, spec, texts):
    if spec.is_vlm:
        return proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device())
    return tokenizer_of(proc, spec)(texts, return_tensors="pt", padding=True, truncation=True).to(device())


def effective_batch_size(batch_size, spec):
    return max(1, batch_size // spec.batch_size_divide)


@torch.no_grad()
def generate_batch(model, proc, spec, texts, max_new_tokens, sampling):
    tok = tokenizer_of(proc, spec)
    inputs = tokenize(proc, spec, texts)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **sampling)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return [text.strip() for text in tok.batch_decode(new_tokens, skip_special_tokens=True)]


def sampling_kwargs(temperature=0.0, top_p=1.0, top_k=None):
    """temperature=0 means greedy decoding."""
    if not temperature:
        return {"do_sample": False}
    kwargs = {"do_sample": True, "temperature": temperature, "top_p": top_p}
    if top_k is not None:
        kwargs["top_k"] = top_k
    return kwargs


def generate(model, proc, spec, prompts, *, max_new_tokens, batch_size, temperature=0.0, top_p=1.0, top_k=None,
             desc="Generating"):
    """One chat-templated completion per prompt, in the order of `prompts`."""
    tok = tokenizer_of(proc, spec)
    texts = [chat_text(tok, spec, p) for p in prompts]
    sampling = sampling_kwargs(temperature, top_p, top_k)
    size = effective_batch_size(batch_size, spec)
    print(f"{spec.name} | batch size {size} | {len(texts)} prompts")
    generations = []
    for batch in tqdm(list(batches(texts, size)), desc=desc):
        generations += generate_batch(model, proc, spec, batch, max_new_tokens, sampling)
    return generations
