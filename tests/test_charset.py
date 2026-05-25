from ocr.charset import DEFAULT_CHARSET


def test_charset_roundtrip():
    text = "Hello 42"
    encoded = DEFAULT_CHARSET.encode(text)
    decoded = DEFAULT_CHARSET.decode(encoded)
    assert decoded == text
