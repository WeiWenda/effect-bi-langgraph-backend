"""Helpers for LangGraph human-in-the-loop interrupts."""


def extract_interrupt_text(value) -> str:
    """Normalize interrupt payload from LangGraph state to user-facing text."""
    if value is None:
        return "Waiting for input."
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [extract_interrupt_text(item) for item in value if item is not None]
        return "\n".join(part for part in parts if part)
    if hasattr(value, "value"):
        return extract_interrupt_text(value.value)
    text = str(value)
    if text.startswith("(") and "Interrupt(value=" in text:
        marker = "Interrupt(value="
        start = text.find(marker)
        if start != -1:
            start += len(marker)
            quote = text[start]
            if quote in ("'", '"'):
                end = text.find(quote, start + 1)
                if end != -1:
                    return text[start + 1 : end]
    return text
