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


def test_current_date_is_supplied_even_when_history_claims_2024():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    messages = build_messages(
        [("Какой год?", "Сейчас 2024 год.")], "А сегодня?", max_turns=12, max_chars=1000
    )
    today = datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat()
    assert today in messages[0]["content"]
    assert "Europe/Moscow" in messages[0]["content"]


@pytest.mark.parametrize(
    "question", ["Какой сейчас год и точная дата?", "какой сейчас год", "Какая сегодня дата?"]
)
def test_current_date_uses_clock_without_model(question):
    from datetime import date

    from telegram_ai.dialogue import clock_answer

    assert clock_answer(question, today=date(2026, 11, 10)) == (
        "Сейчас 2026 год. Точная дата: 10 ноября 2026 года (по Москве)."
    )


@pytest.mark.parametrize(
    "question", ["Какой год был перед 2020?", "Какая сегодня погода?", "Переведи: какой сейчас год"]
)
def test_other_date_mentions_are_not_replaced_by_clock(question):
    from telegram_ai.dialogue import clock_answer

    assert clock_answer(question) is None
