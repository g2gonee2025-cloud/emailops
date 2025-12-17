from cortex.email_processing import clean_email_text


def test_clean_email_text_preserves_quoted_reply_content():
    raw_text = "> Quoted line\nActual response\n>> Nested quote"

    cleaned = clean_email_text(raw_text)

    assert "Quoted line" in cleaned
    assert "Nested quote" in cleaned
    assert ">" not in cleaned
