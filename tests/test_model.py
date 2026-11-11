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
        assert body["reasoning"] == {"max_tokens": 512, "exclude": True}
        assert "models" not in body
        assert body["max_tokens"] == settings.max_output_tokens
        assert body["tool_choice"] == "auto"
        assert [tool["function"]["name"] for tool in body["tools"]] == ["web_search"]
        return web.json_response({"choices": [{"message": {"content": "x" * 20000}}]})

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
            assert body["tool_choice"] == "auto"
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
        assert body["tool_choice"] == "none"
        assert body["tools"][0]["function"]["name"] == "web_search"
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


@pytest.mark.parametrize("extra", [{}, {"tool_calls": None}, {"tool_calls": []}])
async def test_plain_greeting_is_a_valid_answer(settings, extra):
    async def respond(request):
        return web.json_response(
            {"choices": [{"message": {"content": "Привет! Чем могу помочь?", **extra}}]}
        )

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        assert await client.complete([{"role": "user", "content": "привет"}]) == (
            "Привет! Чем могу помочь?"
        )


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


async def test_streamed_tool_arguments_are_reassembled_when_provider_streams(settings, monkeypatch):
    from telegram_ai.search import WebSearch

    async def search(self, query):
        assert query == "latest facts"
        return [{"title": "Source", "url": "https://example.com/", "snippet": "fact", "text": ""}]

    monkeypatch.setattr(WebSearch, "search", search)
    requests = 0
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
                                "function": {"name": "web_search", "arguments": ""},
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
                        "tool_calls": [{"index": 0, "function": {"arguments": '{"query":"latest '}}]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'facts"}'}}]},
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
        nonlocal requests
        requests += 1
        if requests == 2:
            return web.json_response(
                {"choices": [{"message": {"content": "Проверенный факт [1]."}}]}
            )
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
        assert "Проверенный факт" in await c.complete(
            [{"role": "system", "content": "current date"}]
        )


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


async def test_multiple_search_calls_have_matching_results_and_one_lookup(settings, monkeypatch):
    from telegram_ai.search import WebSearch

    searches = []

    async def search(self, query):
        searches.append(query)
        return [{"title": "Source", "url": "https://example.com/", "snippet": "fact", "text": ""}]

    monkeypatch.setattr(WebSearch, "search", search)
    requests = []
    calls = [
        {
            "id": f"call{i}",
            "type": "function",
            "function": {"name": "web_search", "arguments": json.dumps({"query": query})},
        }
        for i, query in enumerate(["latest facts", "latest facts", "other facts"])
    ]

    async def respond(request):
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Unverified preamble",
                                "tool_calls": calls,
                            }
                        }
                    ]
                }
            )
        assert body["messages"][-4]["tool_calls"] == calls
        results = body["messages"][-3:]
        assert [r["tool_call_id"] for r in results] == ["call0", "call1", "call2"]
        assert json.loads(results[0]["content"])["sources"]
        assert results[0]["content"] == results[1]["content"]
        assert "error" in json.loads(results[2]["content"])
        return web.json_response({"choices": [{"message": {"content": "Проверенный факт [1]."}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        answer = await client.complete([{"role": "system", "content": "current date"}])
        assert searches == ["latest facts"]
        assert "Unverified preamble" not in answer
        assert "https://example.com/" in answer


@pytest.mark.parametrize("content", [None, "", "   ", 42, "<tool_call>bad</tool_call>"])
async def test_invalid_content_is_not_sent_to_user(settings, content):
    async def respond(request):
        return web.json_response({"choices": [{"message": {"content": content}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        with pytest.raises(ModelError) as failure:
            await client.complete([])
        assert failure.value.kind == "invalid_response"


async def test_current_date_does_not_call_provider(settings, monkeypatch):
    async def fail(*args, **kwargs):
        pytest.fail("Clock question must not depend on model or search availability")

    async with ClientSession() as session:
        client = ModelClient(session, settings)
        monkeypatch.setattr(client, "_request", fail)
        monkeypatch.setattr(client.search, "search", fail)
        answer = await client.complete(
            [{"role": "user", "content": "Какой сейчас год и точная дата?"}]
        )
        from datetime import datetime
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("Europe/Moscow")).date()
        assert str(today.year) in answer
        assert "по Москве" in answer


@pytest.mark.parametrize(
    "calls",
    [
        "not a list",
        [{"id": "a", "type": "function", "function": {"name": "unknown", "arguments": "{}"}}],
        [{"id": "a", "type": "function", "function": {"name": "web_search", "arguments": "{"}}],
        [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query":42}'},
            }
        ],
        [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query":"news"}'},
            }
        ]
        * 2,
    ],
)
async def test_malformed_calls_are_rejected_before_search(settings, monkeypatch, calls):
    async def respond(request):
        return web.json_response({"choices": [{"message": {"tool_calls": calls}}]})

    async def no_search(query):
        pytest.fail("Invalid calls must not trigger external search")

    app = web.Application()
    app.router.add_post("/chat/completions", respond)
    async with TestServer(app) as server, ClientSession() as session:
        client = ModelClient(session, replace(settings, llm_api_base=str(server.make_url(""))))
        monkeypatch.setattr(client.search, "search", no_search)
        with pytest.raises(ModelError) as failure:
            await client.complete([])
        assert failure.value.kind == "invalid_response"
