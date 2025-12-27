from cortex.common.models import SecureBaseModel


class PiiModel(SecureBaseModel):
    _PII_FIELDS: set[str] = {"secret"}
    secret: str
    public: str


def test_secure_base_model_redacts_pii():
    """Verify that the SecureBaseModel redacts fields listed in _PII_FIELDS."""

    model = PiiModel(secret="password", public="hello")

    assert "secret='*****'" in repr(model)
    assert "public='hello'" in repr(model)
