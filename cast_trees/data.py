"""Prompt sets and chat encoding."""
import torch
from datasets import load_dataset

DOMAINS = ("gsm8k", "mtbench", "humaneval", "math500", "mbpp")
BOXED = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE = "Write a solution to the following problem and make sure that it passes the tests:\n"


def load_prompts(domain, n=None):
    """First `n` prompts of a domain, in dataset order."""
    if domain == "gsm8k":
        prompts = [r["question"] + BOXED for r in load_dataset("openai/gsm8k", "main", split="test")]
    elif domain == "mtbench":
        # first turn only
        prompts = [r["prompt"][0] for r in load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")]
    elif domain == "humaneval":
        prompts = [CODE + "```python\n" + r["prompt"] + "\n```"
                   for r in load_dataset("openai/openai_humaneval", split="test")]
    elif domain == "math500":
        prompts = [r["problem"] + BOXED for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    elif domain == "mbpp":
        prompts = [(r.get("text") or r.get("prompt")) + "\nYour code should pass these tests:\n```python\n"
                   + "\n".join(r["test_list"]) + "\n```"
                   for r in load_dataset("google-research-datasets/mbpp", "mbpp", split="test")]
    else:
        raise ValueError(f"unknown domain {domain}")
    return prompts[:n] if n else prompts


def encode(tokenizer, prompt, device="cuda"):
    """Single user turn with the chat template (thinking off for Qwen3)."""
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        add_generation_prompt=True, enable_thinking=False,
                                        return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids.to(device)


def stop_ids(tokenizer):
    ids = [tokenizer.eos_token_id]
    if "<|eot_id|>" in tokenizer.get_vocab():    # LLaMA-3 end of turn
        ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    return ids
