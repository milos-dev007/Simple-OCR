from ocr.charset import DEFAULT_CHARSET, LOWERCASE_CHARSET, get_charset


def test_charset_roundtrip():
    text = "Hello 42"
    encoded = DEFAULT_CHARSET.encode(text)
    decoded = DEFAULT_CHARSET.decode(encoded)
    assert decoded == text


def test_lowercase_charset_rejects_uppercase_and_digits():
    assert LOWERCASE_CHARSET.contains("hello")
    assert not LOWERCASE_CHARSET.contains("Hello 42")
    assert get_charset("lowercase") is LOWERCASE_CHARSET
