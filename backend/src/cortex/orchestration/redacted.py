"""Custom Pydantic type for redacted fields."""


class Redacted(str):
    """
    A string that should be redacted in logs.
    By subclassing str, Pydantic can validate it as a string.
    The custom __repr__ in the GraphState model will handle redaction.
    """

    pass
