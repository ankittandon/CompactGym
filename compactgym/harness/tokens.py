"""Exact context token counting + rendering in the main model's (gpt-oss) harmony format.

The Fireworks chat template differs from harmony by a small constant (calibrated by scripts/verify.py), so
counts here are lower bounds off by ``Config.prompt_overhead_tokens``.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyEncodingName, Message as HMessage,
                            ReasoningEffort, Role, SystemContent, ToolDescription, load_harmony_encoding)

from ..interfaces import Message


@lru_cache(maxsize=1)
def encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _tool_descs(tools: list[dict]) -> list[ToolDescription]:
    return [ToolDescription.new(t["function"]["name"], t["function"].get("description", ""), parameters=t["function"].get("parameters")) for t in tools]


def to_harmony(messages: list[Message], tools: list[dict]) -> list[HMessage]:
    """Our messages -> harmony messages. System prompt(s) become developer instructions (+ tool defs)."""
    sys_text = "\n\n".join(m.content or "" for m in messages if m.role == "system")
    dev = DeveloperContent.new().with_instructions(sys_text)
    if tools:
        dev = dev.with_function_tools(_tool_descs(tools))
    out = [HMessage.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)),
           HMessage.from_role_and_content(Role.DEVELOPER, dev)]
    names: dict[str, str] = {}
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            out.append(HMessage.from_role_and_content(Role.USER, m.content or ""))
        elif m.role == "assistant":
            if m.content and m.tool_calls:
                out.append(HMessage.from_role_and_content(Role.ASSISTANT, m.content).with_channel("commentary"))
            elif m.content or not m.tool_calls:
                out.append(HMessage.from_role_and_content(Role.ASSISTANT, m.content or "").with_channel("final"))
            for tc in m.tool_calls or []:
                fn = tc["function"]
                names[tc["id"]] = fn["name"]
                args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                out.append(HMessage.from_role_and_content(Role.ASSISTANT, args).with_channel("commentary")
                           .with_recipient(f"functions.{fn['name']}").with_content_type("json"))
        elif m.role == "tool":
            nm = names.get(m.tool_call_id or "", m.meta.get("name", "unknown"))
            out.append(HMessage.from_author_and_content(Author.new(Role.TOOL, f"functions.{nm}"), m.content or "").with_channel("commentary"))
    return out


def render_prompt(messages: list[Message], tools: list[dict]) -> list[int]:
    return list(encoding().render_conversation_for_completion(Conversation.from_messages(to_harmony(messages, tools)), Role.ASSISTANT))


TOOL_MSG_OVERHEAD = 5  # Fireworks' chat template adds exactly 5 tokens per tool message vs harmony (fit with zero residual, 1184 calls)


def count_tokens(messages: list[Message], tools: list[dict], overhead: int = 0, per_tool: int | None = None) -> int:
    """Exact prompt-token count as the API reports it: harmony render + ``overhead`` (constant header gap, 11) + 5 per tool message.
    With ``overhead=0`` (the default) the raw harmony count is returned."""
    if per_tool is None:
        per_tool = TOOL_MSG_OVERHEAD if overhead else 0
    return len(render_prompt(messages, tools)) + overhead + per_tool * sum(m.role == "tool" for m in messages)


def text_tokens(text: str) -> int:
    return len(_tok().encode(text, allowed_special="all"))


_msg_cache: dict[str, int] = {}


def message_tokens(m: Message) -> int:
    """Cheap per-message estimate (content + tool-call JSON + ~4 framing tokens), cached by content hash."""
    key = hashlib.md5(json.dumps([m.role, m.content, m.tool_calls], sort_keys=True, default=str).encode()).hexdigest()
    n = _msg_cache.get(key)
    if n is None:
        enc = _tok()
        txt = (m.content or "") + (json.dumps(m.tool_calls) if m.tool_calls else "")
        n = len(enc.encode(txt, allowed_special="all")) + 4
        _msg_cache[key] = n
    return n


@lru_cache(maxsize=1)
def _tok():
    import tiktoken
    try:
        return tiktoken.get_encoding("o200k_harmony")
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def msg_hash(m: Message) -> str:
    return hashlib.md5(json.dumps([m.role, m.content, m.tool_calls, m.tool_call_id], sort_keys=True, default=str).encode()).hexdigest()
