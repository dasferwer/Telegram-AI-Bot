import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import aiohttp

from .settings import Settings


@dataclass
class ModelError(Exception):
    kind: str
    message: str
    retryable: bool = False
    retry_after: float = 3

    def __str__(self) -> str:
        return self.message


def model_error(status: int, retry_after: float = 3) -> ModelError:
    if status in {402, 429}:
        return ModelError(
            "quota",
            "Лимит бесплатного API исчерпан или временно превышен. "
            "Попробуйте позже. Платная модель не подключается.",
        )
    if status in {401, 403}:
        return ModelError(
            "configuration",
            "Провайдер отклонил доступ к модели. "
            "Нужна проверка ключа, настроек доступа или региона.",
        )
    return ModelError(
        "unavailable",
        "Модель сейчас недоступна. Попробуйте позже.",
        retryable=status >= 500 and retry_after <= 60,
        retry_after=retry_after,
    )


class ModelClient:
    def __init__(self, session: aiohttp.ClientSession, settings: Settings):
        self.session, self.settings = session, settings

    async def verify_free_model(self) -> None:
        if not self.settings.llm_free_only:
            return
        async with self.session.get(
            f"{self.settings.llm_api_base}/models",
            timeout=aiohttp.ClientTimeout(total=20),
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise RuntimeError("Не удалось проверить бесплатный каталог моделей")
            data = await response.json()
        for model in data.get("data", []):
            if model.get("id") != self.settings.llm_model:
                continue
            try:
                pricing = model["pricing"]
                if {"prompt", "completion"} <= pricing.keys() and all(
                    Decimal(str(pricing.get(key, "0"))) == 0
                    for key in (
                        "prompt",
                        "completion",
                        "request",
                        "internal_reasoning",
                        "input_cache_read",
                        "input_cache_write",
                    )
                ):
                    return
            except (KeyError, InvalidOperation, TypeError, ValueError):
                pass
            raise RuntimeError("У модели обнаружена ненулевая или неизвестная цена")
        raise RuntimeError("Бесплатная модель отсутствует в каталоге")

    async def complete(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": self.settings.max_output_tokens,
            "stream": False,
        }
        if self.settings.llm_free_only:
            body["reasoning"] = {"enabled": False}
            body["provider"] = {
                "max_price": {"prompt": 0, "completion": 0},
                "allow_fallbacks": False,
            }
        try:
            async with self.session.post(
                f"{self.settings.llm_api_base}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.settings.model_timeout, connect=10),
                allow_redirects=False,
            ) as response:
                try:
                    retry_after = max(1, float(response.headers.get("Retry-After", "3")))
                except ValueError:
                    retry_after = 3
                if response.status != 200:
                    raise model_error(response.status, retry_after)
                # Bound response memory even if a provider ignores max_tokens.
                raw = bytearray()
                async for chunk in response.content.iter_chunked(16384):
                    raw.extend(chunk)
                    if len(raw) > 512000:
                        raise ModelError(
                            "invalid_response", "Модель вернула слишком большой ответ."
                        )
                data = json.loads(raw)
                if "error" in data:
                    raise model_error(int(data["error"].get("code", 502)), retry_after)
                text = data["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty answer")
                text = text.strip()
                text.encode("utf-8")
                if len(text) > self.settings.max_output_chars:
                    note = "\n[Ответ сокращён по ограничению длины.]"
                    if len(note) < self.settings.max_output_chars:
                        text = text[: self.settings.max_output_chars - len(note)] + note
                    else:
                        text = text[: self.settings.max_output_chars]
                return text
        except TimeoutError:
            raise ModelError(
                "timeout",
                "Модель не ответила за отведённое время. Попробуйте позже.",
                retryable=True,
            ) from None
        except aiohttp.ClientError:
            raise ModelError(
                "network", "Не удалось связаться с моделью. Попробуйте позже.", retryable=True
            ) from None
        except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
            raise ModelError(
                "invalid_response", "Модель вернула некорректный ответ. Попробуйте позже."
            ) from None
