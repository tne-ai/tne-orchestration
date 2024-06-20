from typing import List, TypedDict, Literal

from jinja2 import Environment, BaseLoader

Message = TypedDict('Message', {
    "role": Literal["system", "user", "assistant"],
    "content": str,
})

# Llama3 ChatFormat
# https://github.com/meta-llama/llama3/blob/71367a6cafecf2eea125c374161e6cea10b69dfa/llama/tokenizer.py#L202
_LLAMA3_PROMPT_TEMPLATE = """\
<|begin_of_text|>{%- for m in messages -%}<|start_header_id|>{{ m['role'] }}<|end_header_id|>

{{ m['content'] }}<|eot_id|>{%- endfor -%}
<|start_header_id|>assistant<|end_header_id|>

"""


def apply_chat_template(model: str, messages: List[Message]) -> str:
    """
    Converts a list of dictionaries with "role" and "content" keys to instruction formatted message.

    Args:
        model: The model name.
        messages: A list of messages.

    Returns:
        str: The formatted message.
    """

    model = model.lower()
    if model.startswith("llama-3-"):
        env = Environment(loader=BaseLoader()).from_string(_LLAMA3_PROMPT_TEMPLATE)
        return env.render(messages=messages)
    else:
        raise ValueError("unsupported model")
