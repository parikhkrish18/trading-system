import pytest
from cryptography.fernet import Fernet

from execution import client_crypto


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setattr(client_crypto.settings, "client_key_encryption_key", Fernet.generate_key().decode())


def test_encrypt_decrypt_round_trips():
    ciphertext = client_crypto.encrypt_credential("PKTESTKEY123")
    assert client_crypto.decrypt_credential(ciphertext) == "PKTESTKEY123"


def test_ciphertext_is_not_the_plaintext():
    ciphertext = client_crypto.encrypt_credential("super-secret-alpaca-key")
    assert b"super-secret-alpaca-key" not in ciphertext


def test_encrypt_refuses_without_an_encryption_key(monkeypatch):
    monkeypatch.setattr(client_crypto.settings, "client_key_encryption_key", "")
    with pytest.raises(RuntimeError, match="CLIENT_KEY_ENCRYPTION_KEY"):
        client_crypto.encrypt_credential("anything")


def test_decrypt_raises_clearly_when_the_key_was_rotated(monkeypatch):
    ciphertext = client_crypto.encrypt_credential("a-client-secret")
    monkeypatch.setattr(client_crypto.settings, "client_key_encryption_key", Fernet.generate_key().decode())
    with pytest.raises(RuntimeError, match="Could not decrypt"):
        client_crypto.decrypt_credential(ciphertext)


def test_hash_password_round_trips():
    stored = client_crypto.hash_password("correct horse battery staple")
    assert client_crypto.verify_password("correct horse battery staple", stored) is True


def test_verify_password_rejects_wrong_password():
    stored = client_crypto.hash_password("the-real-password")
    assert client_crypto.verify_password("a-guess", stored) is False


def test_hash_password_is_salted_differently_each_time():
    a = client_crypto.hash_password("same-password")
    b = client_crypto.hash_password("same-password")
    assert a != b
    assert client_crypto.verify_password("same-password", a) is True
    assert client_crypto.verify_password("same-password", b) is True


def test_verify_password_handles_a_malformed_stored_hash():
    assert client_crypto.verify_password("anything", "not-a-real-hash") is False
    assert client_crypto.verify_password("anything", "") is False
