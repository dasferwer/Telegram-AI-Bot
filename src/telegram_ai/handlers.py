from aiogram import Dispatcher, Router
from aiogram.types import Message, Update

from .dialogue import BUSY, HELP, NEW, ONLY_TEXT, RATE_LIMIT, START, access_denied
from .settings import Settings
from .storage import Store
from .worker import Workers


def create_dispatcher() -> Dispatcher:
    router = Router()

    @router.message()
    async def receive(
        message: Message,
        event_update: Update,
        store: Store,
        workers: Workers,
        settings: Settings,
    ) -> dict | None:
        user = message.from_user
        # The HTTP entry point also checks access; direct dispatcher use remains closed.
        if message.chat.type != "private" or user is None:
            return None
        if user.id not in settings.allowed_user_ids:
            return {
                "method": "sendMessage",
                "chat_id": message.chat.id,
                "text": access_denied(user.id),
            }
        text = message.text
        command = text.split(maxsplit=1)[0].split("@", 1)[0] if text else ""
        options = {}
        if message.content_type != "text":
            options["answer"] = ONLY_TEXT
        elif command == "/start":
            options["answer"] = START
        elif command == "/help":
            options["answer"] = HELP
        elif command == "/new":
            options.update(answer=NEW, reset=True)
        elif command.startswith("/"):
            options["answer"] = "Неизвестная команда. Список команд: /help."
        elif not text or not text.strip():
            options["answer"] = "Отправьте непустое текстовое сообщение."
        elif len(text) > settings.max_input_chars:
            options["answer"] = f"Сократите сообщение до {settings.max_input_chars} символов."
        else:
            try:
                text.encode("utf-8")
            except UnicodeError:
                options["answer"] = (
                    "Не удалось прочитать символы сообщения. Отправьте текст ещё раз."
                )
            else:
                options["question"] = text
        result = await store.accept(event_update.update_id, user.id, **options)
        if result == "accepted":
            workers.wake()
        if result in {"busy", "rate"}:
            return {
                "method": "sendMessage",
                "chat_id": message.chat.id,
                "text": BUSY if result == "busy" else RATE_LIMIT,
            }
        return None

    dispatcher = Dispatcher(disable_fsm=True)
    dispatcher.include_router(router)
    return dispatcher
