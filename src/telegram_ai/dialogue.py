SYSTEM_MESSAGE = (
    "Ты собеседник в Telegram. По умолчанию отвечай по-русски, "
    "но используй язык пользователя, если это уместно или он об этом просит. "
    "Отвечай кратко и по существу. Используй обычный текст без HTML-разметки."
)

START = (
    "Здесь можно переписываться с нейросетью. Сообщения и часть истории передаются "
    "OpenRouter и внешнему провайдеру модели, а история хранится в облачной БД. "
    "Не отправляйте секреты и чувствительные данные. "
    "Доступны только текстовые сообщения. Команды: /help и /new."
)
HELP = (
    "/start — описание и обработка данных\n"
    "/help — команды и ограничения\n"
    "/new — удалить историю и отменить незавершённые ответы\n\n"
    "Работаю только в личных чатах и только с текстом. "
    "Бесплатные сервисы имеют квоты и могут быть недоступны. "
    "После простоя первый ответ может задержаться на минуту и больше."
)
NEW = "История очищена. Начинаем новый диалог."
ONLY_TEXT = "Пока доступен только текст. Отправьте сообщение без вложений."
BUSY = "Слишком много сообщений. Дождитесь ответа и отправьте сообщение ещё раз."
RATE_LIMIT = "Сообщения поступают слишком часто. Подождите несколько секунд и повторите."
GENERATION_FAILED = "Не удалось получить ответ после повторных попыток. Отправьте вопрос ещё раз."


def access_denied(user_id: int) -> str:
    return f"Доступ закрыт. Ваш Telegram user ID: {user_id}."


def build_messages(
    turns: list[tuple[str, str]],
    question: str,
    *,
    max_turns: int,
    max_chars: int,
) -> list[dict[str, str]]:
    if len(question) > max_chars:
        raise ValueError("Question exceeds the context budget")
    remaining = max_chars - len(question)
    selected: list[tuple[str, str]] = []
    for user, assistant in reversed(turns[-max_turns:]):
        size = len(user) + len(assistant)
        if size > remaining:
            break
        selected.append((user, assistant))
        remaining -= size
    result = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for user, assistant in reversed(selected):
        result.extend(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        )
    result.append({"role": "user", "content": question})
    return result


def split_text(text: str, limit: int = 4000) -> list[str]:
    chunks: list[str] = []
    start = units = 0
    for index, char in enumerate(text):
        size = 2 if ord(char) > 0xFFFF else 1
        if units + size > limit:
            chunks.append(text[start:index])
            start, units = index, 0
        units += size
    if start < len(text):
        chunks.append(text[start:])
    return chunks
