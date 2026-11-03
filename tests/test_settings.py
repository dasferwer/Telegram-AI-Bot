import pytest

from telegram_ai.settings import Settings


def test_empty_allowlist_is_closed():
    settings = Settings.from_env({})
    assert not settings.allowed_user_ids


@pytest.mark.parametrize("model", ["google/gemma-4-31b-it", "openrouter/auto"])
def test_paid_or_automatic_model_is_rejected(model):
    with pytest.raises(ValueError, match="free"):
        Settings.from_env({"LLM_MODEL": model})


def test_invalid_user_id_is_rejected():
    with pytest.raises(ValueError):
        Settings.from_env({"ALLOWED_USER_IDS": "1,not-an-id"})


def test_neon_direct_url_is_loaded_without_exposing_credentials_in_repr():
    settings = Settings.from_env(
        {
            "DATABASE_URL": "postgresql://user:secret@pooled.example/db",
            "DATABASE_URL_UNPOOLED": "postgresql://user:secret@direct.example/db",
        }
    )
    assert settings.database_url_unpooled == "postgresql://user:secret@direct.example/db"
    assert "secret" not in repr(settings)
