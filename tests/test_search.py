import socket

import pytest

from telegram_ai.search import PublicResolver, WebSearch, public_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a",
        "http://[::1]/",
        "http://169.254.169.254/latest",
        "http://10.0.0.1/",
        "https://localhost/",
        "file:///etc/passwd",
        "https://user:pass@example.com/",
        "https://example.com:8080/",
        "http://site.local/",
    ],
)
def test_private_and_non_web_urls_are_rejected(url):
    assert public_url(url) is None


async def test_dns_rebinding_to_private_address_is_rejected(monkeypatch):
    async def resolve(*args, **kwargs):
        return [{"host": "127.0.0.1"}]

    resolver = PublicResolver()
    monkeypatch.setattr(resolver.inner, "resolve", resolve)
    try:
        with pytest.raises(OSError):
            await resolver.resolve("public.example", 443, socket.AF_INET)
    finally:
        await resolver.close()


async def test_search_returns_bounded_sources_and_no_private_targets(monkeypatch):
    search = WebSearch()
    monkeypatch.setattr(
        search,
        "_lookup",
        lambda q: [
            {"title": "unsafe", "href": "http://127.0.0.1/", "body": "ignore"},
            {"title": "Source", "href": "https://example.com/a", "body": "snippet"},
            {"title": "Duplicate", "href": "https://example.com/a", "body": "again"},
        ],
    )

    async def fetch(url):
        return "page content " * 3000

    monkeypatch.setattr(search, "_fetch", fetch)
    result = await search.search("latest facts")
    assert len(result) == 1
    assert result[0]["url"] == "https://example.com/a"
    assert "snippet" in result[0]["snippet"]
    assert len(result[0]["text"]) <= 6000


async def test_search_failure_has_no_fake_results(monkeypatch):
    search = WebSearch()

    def fail(query):
        raise RuntimeError("internal service details")

    monkeypatch.setattr(search, "_lookup", fail)
    assert await search.search("latest facts") == []


async def test_page_redirect_to_private_network_is_not_followed(monkeypatch):
    import aiohttp

    visited = []

    class Response:
        status = 302
        headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def __init__(self, **kwargs):
            self.connector = kwargs["connector"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.connector.close()

        def get(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            visited.append(url)
            return Response()

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    assert await WebSearch()._fetch("https://example.com/") == ""
    assert visited == ["https://example.com/"]


def test_page_extraction_discards_scripts_and_navigation():
    from telegram_ai.search import PageText

    parser = PageText()
    parser.feed("<nav>Menu</nav><script>bad()</script><p>Useful facts</p>")
    assert " ".join(parser.parts) == "Useful facts"
