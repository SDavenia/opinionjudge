"""Llama-Guard-3-8B unsafe score of each opinion: probability of "unsafe" vs "safe" at the
label token of its (greedy) answer."""
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_utils import feature_dir, load_opinions

MODEL_ID = "meta-llama/Llama-Guard-3-8B"


def output_path(dataset_id, args):
    return feature_dir(args.features_dir / "safety", dataset_id, args.generation_model_id) / "opinion_scores.csv"


def done(dataset_id, args):
    return output_path(dataset_id, args).exists()


class LlamaGuardScorer:
    def __init__(self, model_id=MODEL_ID):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        # "safe" / "unsafe" always follow a newline, so their no-leading-space tokens are fixed.
        self.safe_id, self.unsafe_id = self._token_id("safe"), self._token_id("unsafe")

    def _token_id(self, word):
        matches = [tok_id for tok_str, tok_id in self.tokenizer.get_vocab().items()
                   if self.tokenizer.convert_tokens_to_string([tok_str]) == word]
        if len(matches) != 1:
            raise ValueError(f"Expected one token for {word!r}, found {matches}.")
        return matches[0]

    @torch.no_grad()
    def score(self, text, role="user"):
        enc = self.tokenizer.apply_chat_template([{"role": role, "content": text}], add_generation_prompt=True,
                                                 return_tensors="pt", return_dict=True).to(self.device)
        input_ids = enc["input_ids"]
        out = self.model.generate(input_ids, attention_mask=enc["attention_mask"], max_new_tokens=8, do_sample=False,
                                  pad_token_id=self.tokenizer.eos_token_id, output_scores=True,
                                  return_dict_in_generate=True)
        new_tokens = out.sequences[0, input_ids.shape[-1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # The label comes after some formatting tokens: read the logits where it is generated.
        label_step = next((i for i, t in enumerate(new_tokens.tolist()) if t in (self.safe_id, self.unsafe_id)), None)
        if label_step is None:
            raise RuntimeError(f"No safe/unsafe token in the first {len(new_tokens)} tokens: {raw_output!r}")
        logits = out.scores[label_step][0]
        probs = torch.softmax(torch.stack([logits[self.safe_id], logits[self.unsafe_id]]).float(), dim=0)
        label = "unsafe" if new_tokens[label_step].item() == self.unsafe_id else "safe"
        return {"unsafe_score": probs[1].item(), "label": label, "raw_output": raw_output}


def load(args):
    return LlamaGuardScorer()


def extract(scorer, dataset_id, args):
    df = load_opinions(dataset_id, args.generation_model_id)
    if args.limit is not None:
        df = df.head(args.limit)
    rows = [{"text_id": text_id, **scorer.score(text)}
            for text_id, text in tqdm(list(zip(df["text_id"], df["text"])), desc=f"Llama-Guard ({dataset_id})")]
    path = output_path(dataset_id, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")
