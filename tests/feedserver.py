"""A tiny local feed server for live-mode tests: fixtures with relative dates,
ETag / 304 support, and routes that fail on purpose."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "feeds"
_TOKEN = re.compile(r"@@(rfc|iso):([+-]\d+)([hd])@@")


def render(name: str, now: datetime | None = None) -> bytes:
    """Load a fixture and replace @@rfc:-2h@@ / @@iso:-3d@@ with dates relative to now."""
    now = now or datetime.now(timezone.utc)

    def swap(match: re.Match) -> str:
        kind, amount, unit = match.group(1), int(match.group(2)), match.group(3)
        when = now + (timedelta(hours=amount) if unit == "h" else timedelta(days=amount))
        return format_datetime(when) if kind == "rfc" else when.isoformat(timespec="seconds")

    return _TOKEN.sub(swap, (FIXTURES / name).read_text(encoding="utf-8")).encode("utf-8")


class FeedServer:
    """Serves ``routes`` (path -> (status, body, etag)). Mutate ``routes`` to change a feed."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes, str | None]] = {
            "/world.xml": (200, render("world.xml"), '"world-v1"'),
            "/tech.atom": (200, render("tech.atom"), None),
            "/hindi.xml": (200, render("hindi.xml"), '"hindi-v1"'),
            "/broken": (200, (FIXTURES / "broken.html").read_bytes(), None),
            "/gone": (404, b"not found", None),
        }
        self.hits: dict[str, int] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.hits[self.path] = server.hits.get(self.path, 0) + 1
                status, body, etag = server.routes.get(self.path, (404, b"", None))
                if etag and self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.end_headers()
                    return
                self.send_response(status)
                self.send_header("Content-Type", "application/xml; charset=utf-8")
                if etag:
                    self.send_header("ETag", etag)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> FeedServer:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def feeds_toml(self, path: Path, include_failing: bool = True) -> Path:
        feeds = [
            ("world", "/world.xml", "Meridian Wire", "english", "World", 1),
            ("tech", "/tech.atom", "Circuit Notes", "english", "Technology", 2),
            ("hindi", "/hindi.xml", "Samachar Darpan", "hindi", "Top News", 1),
        ]
        if include_failing:
            feeds += [
                ("broken", "/broken", "Broken Gazette", "english", "India", 3),
                ("gone", "/gone", "Gone Daily", "english", "India", 3),
            ]
        blocks = [
            f'[[feed]]\nid = "{i}"\nurl = "{self.url}{p}"\noutlet = "{o}"\nlanguage = "{lang}"\n'
            f'category = "{c}"\nreach = {r}\n'
            for i, p, o, lang, c, r in feeds
        ]
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path
