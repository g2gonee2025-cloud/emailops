from typing import Any

def scrub_json_string(json_string: str) -> str:
    """
    Remove null characters from a JSON string.
    """
    return json_string.replace('\\u0000', '')

def scrub_json(data: Any) -> Any:
    """
    Recursively remove null characters from JSON data.
    """
    if isinstance(data, dict):
        return {k: scrub_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [scrub_json(elem) for elem in data]
    if isinstance(data, str):
        return scrub_json_string(data)
    return data
