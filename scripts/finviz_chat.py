import os
from typing import Dict, List
from openai import OpenAI


def generate_system_prompt(snapshot: Dict[str, str], ticker: str) -> str:
    """Create a concise system prompt that provides the snapshot indicators to the assistant.

    The assistant should treat these indicators as factual, up-to-date context when answering.
    """
    lines = [f"Ticker: {ticker}"]
    if not snapshot:
        lines.append("No snapshot data available.")
    else:
        # keep it compact — one key per line
        for k, v in snapshot.items():
            lines.append(f"{k}: {v}")

    prompt = (
        "You are a helpful financial assistant. Use the following latest Finviz snapshot indicators "
        "as factual context when answering investor questions. Be concise, avoid giving investment "
        "advice that requires licensing, and explain what each indicator means when the user asks.\n\n"
        + "\n".join(lines)
    )
    return prompt


def chat_with_snapshot(messages: List[Dict[str, str]], snapshot: Dict[str, str], ticker: str, api_key: str = None, model: str = "gpt-4.1-nano-2025-04-14") -> str:
    """Send a chat completion request including the snapshot as a system message.

    messages: list of dicts like {"role": "user", "content": "..."}
    Returns assistant text or raises Exception.
    """
    # Prefer explicit api_key argument, then OPENAI_API_KEY environment variable
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY in the environment.")

    # Use OpenAI official client (no OpenRouter base_url). Allow optional OPENAI_API_BASE override.
    base_url = os.environ.get('OPENAI_API_BASE')
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    # Allow model override via env var OPENAI_MODEL
    model = os.environ.get('OPENAI_MODEL', model)

    system_prompt = generate_system_prompt(snapshot, ticker)
    payload = [{"role": "system", "content": system_prompt}] + messages

    resp = client.chat.completions.create(model=model, messages=payload, max_tokens=500, temperature=0.2)
    return resp.choices[0].message.content.strip()
