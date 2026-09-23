def strip_code_fence(text):
    """Strip a leading/trailing ``` or ```json fence some models wrap JSON responses in."""
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if '\n' in text:
            first_line, rest = text.split('\n', 1)
            text = rest if first_line.strip().lower() in ('json', '') else text
    return text.strip()
