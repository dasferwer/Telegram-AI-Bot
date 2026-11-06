import asyncio
import ipaddress
import logging
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import aiohttp
from ddgs import DDGS

logger = logging.getLogger("telegram_ai")
SEARCH_FAILED = (
    "Не удалось получить источники из интернета. "
    "Свежие сведения не проверены. Попробуйте повторить вопрос позже."
)


def public_url(value: str) -> str | None:
    try:
        if len(value) > 2048:
            return None
        parts = urlsplit(value)
        host = (parts.hostname or "").lower().rstrip(".")
        if (
            parts.scheme not in {"http", "https"}
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.port not in {None, 80, 443}
            or any(ord(c) < 33 for c in value)
            or host == "localhost"
            or host.endswith((".local", ".localhost", ".internal"))
        ):
            return None
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global or address.is_multicast or address.is_reserved:
                return None
        except ValueError:
            if "." not in host:
                return None
        return value
    except (ValueError, TypeError):
        return None


class PublicResolver(aiohttp.abc.AbstractResolver):
    def __init__(self) -> None:
        self.inner = aiohttp.resolver.DefaultResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> list:
        records = await self.inner.resolve(host, port, family)
        addresses = [ipaddress.ip_address(r["host"]) for r in records]
        if not addresses or any(
            not a.is_global or a.is_multicast or a.is_reserved for a in addresses
        ):
            raise OSError("Non-public address")
        return records

    async def close(self) -> None:
        await self.inner.close()


class PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "header", "footer", "form"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "header", "footer", "form"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


class WebSearch:
    def __init__(self) -> None:
        self.slots = asyncio.Semaphore(2)

    def _lookup(self, query: str) -> list[dict]:
        return DDGS(timeout=6).text(
            query, max_results=3, backend="duckduckgo,google", region="ru-ru"
        )

    async def search(self, query: str) -> list[dict[str, str]]:
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            return []
        try:
            async with self.slots:
                results = await asyncio.to_thread(self._lookup, query.strip())
            sources = []
            seen = set()
            for item in results[:6]:
                url = public_url(item.get("href", ""))
                if url and url not in seen:
                    seen.add(url)
                    sources.append(
                        {
                            "title": str(item.get("title", ""))[:250],
                            "url": url,
                            "snippet": str(item.get("body", ""))[:1500],
                            "text": "",
                        }
                    )
                if len(sources) == 3:
                    break
            pages = await asyncio.gather(*(self._fetch(s["url"]) for s in sources))
            for source, page in zip(sources, pages, strict=True):
                source["text"] = page[:6000]
            logger.info("web_search sources=%s pages=%s", len(sources), sum(bool(p) for p in pages))
            return sources
        except Exception as exc:
            logger.info("web_search_failed error=%s", type(exc).__name__)
            return []

    async def _fetch(self, url: str) -> str:
        resolver = PublicResolver()
        try:
            connector = aiohttp.TCPConnector(resolver=resolver, limit=3)
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=7, connect=3),
                headers={"User-Agent": "TelegramTextChat/0.1"},
                trust_env=False,
            ) as session:
                for _ in range(3):
                    if not public_url(url):
                        return ""
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            url = urljoin(url, response.headers.get("Location", ""))
                            continue
                        if response.status != 200 or response.content_type not in {
                            "text/html",
                            "text/plain",
                            "application/xhtml+xml",
                        }:
                            return ""
                        data = bytearray()
                        async for chunk in response.content.iter_chunked(16384):
                            data.extend(chunk)
                            if len(data) > 300000:
                                break
                        text = data.decode(response.charset or "utf-8", errors="replace")
                        if response.content_type == "text/plain":
                            return text[:6000]
                        parser = PageText()
                        parser.feed(text)
                        return " ".join(parser.parts)[:6000]
            return ""
        except (aiohttp.ClientError, TimeoutError, ValueError, OSError, LookupError):
            return ""
        finally:
            await resolver.close()
