from typing import ClassVar

from cortex.common.models import SecureBaseModel


class PiiModel(SecureBaseModel):
    _PII_FIELDS: ClassVar[set[str]] = {"secret"}
    secret: str
    public: str


def test_secure_base_model_redacts_pii():
    """Verify that the SecureBaseModel redacts fields listed in _PII_FIELDS."""

    model = PiiModel(secret="password", public="hello")

    repr_str = repr(model)
    assert "'secret': '*****'" in repr_str
    assert "'public': 'hello'" in repr_str
