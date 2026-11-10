import asyncio
import json
from dataclasses import replace

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from telegram_ai.model import ModelClient, ModelError


@pytest.mark.parametrize(
    "status,kind,retryable",
    [
        (429, "quota", False),
        (402, "quota", False),
        (401, "configuration", False),
        (404, "unavailable", False),
        (503, "unavailable", True),
    ],
)
async def test_http_errors_are_safe_and_have_bounded_retry_policy(
    settings, status, kind, retryable
):
    async def respond(request):
        return web.json_response({"error": {"message": "secret provider text"}}, status=status)

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError) as failure:
            await client.complete([{"role": "user", "content": "тест"}])
        assert failure.value.kind == kind
        assert failure.value.retryable is retryable
        assert "secret" not in str(failure.value)


async def test_timeout_is_reported_without_unbounded_inline_retry(settings):
    calls = 0

    async def slow(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/chat/completions", slow)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(
            session,
            replace(
                settings,
                llm_api_base=str(server.make_url("")),
                model_timeout=0.01,
            ),
        )
        with pytest.raises(ModelError, match="время"):
            await client.complete([])
        assert calls == 1


async def test_request_enforces_zero_price_no_model_fallback_and_output_limit(settings):
    async def respond(request):
        body = await request.json()
        assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        assert body["provider"]["allow_fallbacks"] is False
        assert body["reasoning"] == {"enabled": False}
        assert "models" not in body
        assert body["max_tokens"] == settings.max_output_tokens
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "answer_directly",
                                        "arguments": json.dumps({"answer": "x" * 20000}),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        assert len(await client.complete([])) <= settings.max_output_chars


@pytest.mark.parametrize(
    "pricing,allowed",
    [
        ({"prompt": "0", "completion": "0"}, True),
        ({"prompt": "0", "completion": "0.001"}, False),
        ({"prompt": "0", "completion": "0", "request": "0.01"}, False),
        ({"prompt": "0"}, False),
    ],
)
async def test_catalog_refuses_paid_and_unknown_prices(settings, pricing, allowed):
    async def catalog(request):
        return web.json_response({"data": [{"id": settings.llm_model, "pricing": pricing}]})

    app = web.Application()
    app.router.add_get("/models", catalog)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        if allowed:
            await client.verify_free_model()
        else:
            with pytest.raises(RuntimeError):
                await client.verify_free_model()


async def test_fresh_question_searches_then_answers_with_real_source_links(settings, monkeypatch):
    from telegram_ai.search import WebSearch

    requests = []

    async def search(self, query):
        assert query == "Лига чемпионов 2026 таблица"
        return [
            {
                "title": "UEFA",
                "url": "https://www.uefa.com/",
                "snippet": "Published table",
                "text": "Verified page excerpt",
            }
        ]

    monkeypatch.setattr(WebSearch, "search", search)

    async def respond(request):
        body = await request.json()
        requests.append(body)
        assert "plugins" not in body
        assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0}
        if len(requests) == 1:
            assert body["tool_choice"] == "required"
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query":"Лига чемпионов 2026 таблица"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        assert "tools" not in body
        assert body["messages"][-1]["role"] == "tool"
        assert "Verified page excerpt" in body["messages"][-1]["content"]
        return web.json_response({"choices": [{"message": {"content": "Ответ по источнику [1]."}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        answer = await client.complete([{"role": "user", "content": "Кто лидирует сейчас?"}])
        assert len(requests) == 2
        assert "https://www.uefa.com/" in answer


async def test_search_failure_stops_without_inventing_an_answer(settings, monkeypatch):
    from telegram_ai.search import WebSearch

    calls = 0

    async def search(self, query):
        return []

    monkeypatch.setattr(WebSearch, "search", search)

    async def respond(request):
        nonlocal calls
        calls += 1
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"новости"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError, match="Свежие сведения не проверены"):
            await client.complete([])
        assert calls == 1


async def test_plain_answer_cannot_bypass_required_search_decision(settings):
    async def respond(request):
        return web.json_response({"choices": [{"message": {"content": "Сейчас 2024 год"}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError):
            await client.complete([])


async def test_tool_markup_from_final_answer_is_not_sent_to_user(settings, monkeypatch):
    from telegram_ai.search import WebSearch

    calls = 0

    async def search(self, query):
        return [{"title": "Source", "url": "https://example.com/", "snippet": "fact", "text": ""}]

    monkeypatch.setattr(WebSearch, "search", search)

    async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call1",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query":"news"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        body = await request.json()
        assert body["tool_choice"] == "none"
        return web.json_response(
            {"choices": [{"message": {"content": "<tool_call>bad</tool_call>"}}]}
        )

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError):
            await client.complete([{"role": "system", "content": "current date"}])


async def test_streamed_tool_arguments_are_reassembled_when_provider_streams(settings):
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "type": "function",
                                "function": {"name": "answer_directly", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"answer":"Сейчас '}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '2026 год."}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    data = (
        ": OPENROUTER PROCESSING\n\n"
        + "".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events)
        + "data: [DONE]\n\n"
    )

    async def respond(request):
        assert (await request.json())["stream"] is False
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        raw = data.encode()
        for i in range(0, len(raw), 7):
            await response.write(raw[i : i + 7])
        return response

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        c = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        assert await c.complete([]) == "Сейчас 2026 год."


async def test_streamed_provider_error_is_safe(settings):
    async def respond(request):
        return web.Response(
            text='data: {"error":{"code":503,"message":"secret"}}\n\n',
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        c = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError) as error:
            await c.complete([])
        assert error.value.kind == "unavailable"
        assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
        b"data: {not json}\n\ndata: [DONE]\n\n",
    ],
)
async def test_incomplete_or_malformed_stream_is_not_an_answer(settings, raw):
    async def respond(request):
        return web.Response(body=raw, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        c = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError) as failure:
            await c.complete([])
        assert failure.value.kind == "invalid_response"


async def test_multiple_direct_calls_return_all_distinct_answers(settings):
    async def respond(request):
        calls = [
            {
                "id": str(i),
                "type": "function",
                "function": {
                    "name": "answer_directly",
                    "arguments": json.dumps({"answer": answer}),
                },
            }
            for i, answer in enumerate(
                ["Сейчас 2026 год.", "Точная дата: 10 ноября 2026 года.", "Сейчас 2026 год."]
            )
        ]
        return web.json_response({"choices": [{"message": {"tool_calls": calls}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        assert await client.complete([]) == "Сейчас 2026 год.\nТочная дата: 10 ноября 2026 года."


async def test_search_takes_precedence_over_direct_call_in_same_response(settings, monkeypatch):
    from telegram_ai.search import WebSearch

    async def search(self, query):
        return [
            {"title": "Source", "url": "https://example.com/", "snippet": "Fresh fact", "text": ""}
        ]

    monkeypatch.setattr(WebSearch, "search", search)
    requests = []

    async def respond(request):
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "a",
                                        "type": "function",
                                        "function": {
                                            "name": "answer_directly",
                                            "arguments": '{"answer":"Unverified guess"}',
                                        },
                                    },
                                    {
                                        "id": "b",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query":"latest facts"}',
                                        },
                                    },
                                ]
                            }
                        }
                    ]
                }
            )
        assert body["messages"][-1]["tool_call_id"] == "b"
        return web.json_response({"choices": [{"message": {"content": "Проверенный факт [1]."}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        answer = await client.complete([{"role": "system", "content": "current date"}])
        assert "Unverified guess" not in answer
        assert "Проверенный факт" in answer
        assert "https://example.com/" in answer
