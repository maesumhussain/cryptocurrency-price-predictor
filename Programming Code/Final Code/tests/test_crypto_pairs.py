from crypto_pairs import CRYPTO_PAIRS


def test_crypto_pairs_is_not_empty():
    assert isinstance(CRYPTO_PAIRS, dict)
    assert len(CRYPTO_PAIRS) > 0


def test_crypto_pairs_contains_btc():
    assert "BTC/USD" in CRYPTO_PAIRS
    assert CRYPTO_PAIRS["BTC/USD"] == "BTC-USD"


def test_crypto_pairs_keys_and_values_are_strings():
    for key, value in CRYPTO_PAIRS.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert key.endswith("/USD")