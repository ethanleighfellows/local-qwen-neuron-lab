"""Decode Qwen's native XML tool syntax into OpenAI messages."""
import json
import re
import uuid


def parse_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        body = match[1].strip()
        if body.startswith("{"):
            call = json.loads(body)
            name, arguments = call["name"], call.get("arguments", {})
        else:
            function = re.search(r"<function=([^>]+)>(.*?)</function>", body, re.S)
            if not function:
                raise ValueError("Malformed model tool call")
            name, arguments = function[1], {}
            for param in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", function[2], re.S):
                value = param[2].strip()
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                arguments[param[1]] = value
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}})
    if "<tool_call>" in text and not calls:
        raise ValueError("Incomplete model tool call")
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    result = {"role": "assistant", "content": content or None}
    if calls:
        result["tool_calls"] = calls
    return result
