# -*- coding: utf-8 -*-
"""Bedrock drop-in for model_openai's generate/generate_json.
Same signatures, so `from model_openai import generate, generate_json` keeps working once
model_openai.py delegates here. Uses the AWS Bedrock Converse API with the cross-family
inference-profile models (us. prefix required for cross-region).
Read before adapting: this reimplements ONLY the two public functions the agents import,
matching model_openai.OpenAI_Model.generate's contract (text or metadata dict; JSON variant)."""
import os, re, json, time, random as _rand_mod
_rand = _rand_mod.Random(0)
import boto3
from botocore.config import Config

_REGION = os.environ.get("AWS_REGION", "us-east-1")
_client = boto3.client("bedrock-runtime", region_name=_REGION,
                       config=Config(read_timeout=300, retries={"max_attempts": 8, "mode": "adaptive"}))

# map friendly names -> Bedrock inference-profile ids (us. prefix for cross-region)
_ALIASES = {
    "llama4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "llama4-scout": "us.meta.llama4-scout-17b-instruct-v1:0",
    "llama3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
    "claude-sonnet4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "mistral-pixtral-l": "us.mistral.pixtral-large-2502-v1:0",
    "palmyra-x5": "us.writer.palmyra-x5-v1:0",
    "nova2-lite": "global.amazon.nova-2-lite-v1:0",
}


def _resolve(model):
    if model in _ALIASES:
        return _ALIASES[model]
    if model.startswith(("gpt", "o1", "o3", "ft:")):   # any OpenAI id -> cheap default
        return _ALIASES["llama4-maverick"]
    return model


def format_messages(messages, variables=None):
    variables = variables or {}
    user_msgs = [m for m in messages if m["role"] == "user"]
    if user_msgs and variables:
        last = user_msgs[-1]
        for k, v in variables.items():
            last["content"] = last["content"].replace(f"[[{k}]]", str(v))
    return messages


def _split(messages):
    """Bedrock Converse: system goes in a separate list; roles must alternate user/assistant."""
    system, conv = [], []
    for m in messages:
        if m["role"] == "system":
            system.append({"text": m["content"]})
            continue
        role = "assistant" if m["role"] == "assistant" else "user"
        if conv and conv[-1]["role"] == role:                 # merge consecutive same-role
            conv[-1]["content"][0]["text"] += "\n" + m["content"]
        else:
            conv.append({"role": role, "content": [{"text": m["content"]}]})
    if conv and conv[0]["role"] != "user":                    # Converse must start with user
        conv.insert(0, {"role": "user", "content": [{"text": "(begin)"}]})
    return system, conv


def generate(messages, model="gpt-4o-mini", timeout=30, max_retries=6, temperature=1.0,
             is_json=False, return_metadata=False, max_tokens=None, variables=None):
    messages = format_messages(list(messages), variables or {})
    system, conv = _split(messages)
    if is_json:
        system.append({"text": "Respond with ONLY a single valid JSON object, no prose, no code fences."})
    mid = _resolve(model)
    inf = {"maxTokens": int(max_tokens) if max_tokens else 2048,
           "temperature": float(max(0.0, min(1.0, temperature)))}
    last_err = None
    for attempt in range(max_retries):
        try:
            kw = dict(modelId=mid, messages=conv, inferenceConfig=inf)
            if system:
                kw["system"] = system
            r = _client.converse(**kw)
            text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
            u = r.get("usage", {})
            if not return_metadata:
                return text
            return {"message": text, "total_tokens": u.get("totalTokens", 0),
                    "prompt_tokens": u.get("inputTokens", 0), "prompt_tokens_cached": 0,
                    "completion_tokens": u.get("outputTokens", 0), "total_usd": 0.0}
        except Exception as e:
            last_err = e
            # exponential backoff with jitter - Bedrock throttles hard under concurrency
            time.sleep(min(60, 4 * (2 ** attempt)) * (0.7 + 0.6 * _rand.random()))
    raise Exception(f"Bedrock generate failed ({mid}): {last_err}")


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)   # first balanced-ish object
        if m:
            return json.loads(m.group(0))
        raise


def generate_json(messages, model="gpt-4o-mini", **kwargs):
    kwargs["is_json"] = True
    kwargs.setdefault("return_metadata", True)
    resp = generate(messages, model, **kwargs)
    if isinstance(resp, dict):
        resp["message"] = _extract_json(resp["message"])
        return resp
    return _extract_json(resp)


if __name__ == "__main__":
    print(generate([{"role": "user", "content": "Reply with the single word: ok"}],
                   model="llama4-maverick", return_metadata=True))
