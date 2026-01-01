from typing import Any


def scrub_json_string(json_string: str) -> str:
    """
    Remove null characters from a JSON string.
    """
    if not json_string:
        return json_string
    cleaned = json_string.replace("\x00", "")
    return cleaned.replace("\\u0000", "")


def scrub_json(data: Any) -> Any:
    """
    Recursively remove null characters from JSON data.
    """
    if isinstance(data, dict):
        cleaned: dict[Any, Any] = {}
        for key, value in data.items():
            cleaned_key = scrub_json_string(key) if isinstance(key, str) else key
            cleaned[cleaned_key] = scrub_json(value)
        return cleaned
    if isinstance(data, list):
        return [scrub_json(elem) for elem in data]
    if isinstance(data, str):
        return scrub_json_string(data)
    return data
