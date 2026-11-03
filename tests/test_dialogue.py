import pytest

from telegram_ai.dialogue import build_messages, split_text


def test_split_preserves_text_and_handles_utf16_and_markup():
    text = ("🙂<b>Привет</b>\n" * 1500) + "конец"
    chunks = split_text(text)
    assert "".join(chunks) == text
    assert all(0 < len(part.encode("utf-16-le")) // 2 <= 4000 for part in chunks)


def test_context_keeps_recent_complete_turns_within_character_budget():
    turns = [("старое", "ответ"), ("последнее", "новый ответ")]
    result = build_messages(turns, "вопрос", max_turns=1, max_chars=30)
    assert [message["role"] for message in result] == ["system", "user", "assistant", "user"]
    assert result[1]["content"] == "последнее"
    assert sum(len(message["content"]) for message in result[1:]) <= 30


def test_oversized_question_is_rejected_instead_of_silently_truncated():
    with pytest.raises(ValueError):
        build_messages([], "x" * 31, max_turns=1, max_chars=30)
