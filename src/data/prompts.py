"""Prompt formatting utilities.

Two template styles:
  - ``generate_prompt``: few-shot in-context format used by Experiment 1.
    The buffer phrase ``This statement is:`` decouples the truth signal from
    the prediction token (Marks & Tegmark, 2024).
  - ``apply_chat_template_fn``: model-specific chat template used by
    Experiments 2–5. Detects the architecture from ``tokenizer.name_or_path``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Default few-shot examples for cities (Experiment 1)
DEFAULT_CITY_EXAMPLES: List[Tuple[str, str]] = [
    ("The city of Tokyo is in Japan", "TRUE"),
    ("The city of Hanoi is in Poland", "FALSE"),
    ("The city of Munich is in Germany", "TRUE"),
]

# System prompt used in the chat-template path
TRUTH_SYSTEM_PROMPT = (
    "You are a factual validator. Respond with exactly one word: "
    "'True' if the statement is factually correct, or 'False' if it is incorrect. "
    "Do not provide explanations."
)


def generate_prompt(
    target_statement: str,
    examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Few-shot prompt with the ``This statement is:`` buffer.

    Used by Experiment 1 (activation patching). Spatially decouples the
    truth-judgment computation from the response-token generation, which is
    essential for clean STR-based patching experiments.
    """
    if examples is None:
        examples = DEFAULT_CITY_EXAMPLES
    context = "\n".join(f"{stmt}. This statement is: {label}" for stmt, label in examples)
    return f"{context}\n{target_statement.rstrip('.:;!?')}. This statement is:"

def apply_few_shot_fn(
        instructions: List[str], 
        examples: List[List[Tuple[str, str]]] = None
) -> List[str]:
    return [generate_prompt(stmt, expl) for stmt, expl in zip(instructions, examples)]


def apply_chat_template_fn(tokenizer, instructions: List[str], null_object: bool = False) -> List[str]:
    """Wrap statements in model-specific chat templates.

    Architecture detection is done via ``tokenizer.name_or_path``. Falls back
    to a generic instruction format for unknown models.
    """
    model_path = tokenizer.name_or_path.lower()
    prompts: list[str] = []

    for inst in instructions:
        if null_object:
            formatted = (
                f"{inst}"
            )
        elif "qwen" in model_path:
            formatted = (
                f"<|im_start|>system\n{TRUTH_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\nStatement: {inst}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif "llama-3" in model_path:
            formatted = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{TRUTH_SYSTEM_PROMPT}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\nStatement: {inst}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif "gemma" in model_path:
            formatted = (
                f"<start_of_turn>user\n{TRUTH_SYSTEM_PROMPT}\n\n"
                f"Statement: {inst}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            formatted = f"Instruction: {TRUTH_SYSTEM_PROMPT}\nStatement: {inst}\nAnswer:"
        prompts.append(formatted)
    return prompts


def apply_retain_template(tokenizer, instructions: List[str]) -> List[str]:
    """Format Alpaca instructions through the model's native chat template.

    Different from ``apply_chat_template_fn`` because retain prompts are
    natural Alpaca instructions, not factual statements wrapped in a
    classification system prompt.
    """
    model_path = tokenizer.name_or_path.lower()
    prompts: list[str] = []

    for inst in instructions:
        if "qwen" in model_path:
            formatted = (
                f"<|im_start|>user\n{inst}.<|im_end|>\n<|im_start|>assistant\n"
            )
        elif "llama-3" in model_path:
            formatted = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{inst}.<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif "gemma" in model_path:
            formatted = (
                f"<start_of_turn>user\n{inst}.<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            formatted = f"Instruction: {inst}.\nAnswer:"
        prompts.append(formatted)
    return prompts
