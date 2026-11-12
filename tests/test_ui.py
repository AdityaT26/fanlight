"""Browser tests for the web app. Skipped unless Playwright and Chromium are installed:

pip install playwright && python -m playwright install chromium
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

import uvicorn  # noqa: E402

from fanlight.api import create_app  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server(index, profiles, tmp_path_factory):
    from fanlight.config import Settings
    from fanlight.engine import Engine
    from fanlight.store import Store

    folder = tmp_path_factory.mktemp("ui")
    engine = Engine(
        index, profiles, store=Store(folder / "ui.db"), settings=Settings(state_dir=folder, cohorts=4)
    )
    port = _free_port()
    config = uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="error")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    for _ in range(100):
        if instance.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        executable = os.environ.get("FANLIGHT_CHROME")
        try:
            instance = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
        except Exception as exc:  # browser binary not installed
            pytest.skip(f"Chromium not available: {exc}")
        yield instance
        instance.close()


def _open(browser, url, **context):
    page = browser.new_context(**context).new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(url)
    page.wait_for_selector(".story")
    return page, errors


def test_desktop_feed_renders_and_reacts(browser, server):
    page, errors = _open(browser, server, viewport={"width": 1440, "height": 950})
    assert page.locator(".story").count() == 20
    assert page.locator(".wordmark").inner_text() == "Fanlight"
    assert page.locator("#shape .mix span").count() >= 1
    assert page.locator(".dock").is_hidden()

    page.locator(".story").nth(1).locator(".why").click()
    panel = page.locator(".story").nth(1).locator(".why-panel")
    assert panel.is_visible() and "Match" in panel.inner_text()

    page.locator("#new-reader").click()
    page.fill("input[name=name]", "Test reader")
    page.locator("#dialog-categories .chip").first.click()
    page.locator("#dialog-submit").click()
    page.wait_for_function("document.querySelector('#edition').textContent.includes('Test reader')")

    first = page.locator(".story").first.get_attribute("data-id")
    page.locator(".story").first.locator('[data-action="like"]').click()
    page.wait_for_function(f"document.querySelector('.story')?.dataset.id !== {first!r}")
    assert "1 liked" in page.locator("#reader-card").inner_text()
    assert not errors


def test_phone_layout_uses_sheets(browser, server):
    page, errors = _open(
        browser,
        server,
        viewport={"width": 390, "height": 844},
        is_mobile=True,
        has_touch=True,
        device_scale_factor=2,
    )
    assert page.locator(".dock").is_visible()
    reader_panel = page.locator("#panel-reader")
    assert not reader_panel.is_visible()
    page.locator('.dock button[data-sheet="panel-reader"]').click()
    page.wait_for_function("document.querySelector('#panel-reader').classList.contains('open')")
    page.wait_for_timeout(350)
    assert reader_panel.is_visible()
    page.locator("#scrim").click(position={"x": 20, "y": 20})
    page.wait_for_function("!document.querySelector('#panel-reader').classList.contains('open')")
    # Nothing overflows the viewport sideways.
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    assert not errors


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    from feedserver import FeedServer

    from fanlight.config import Settings
    from fanlight.engine import Engine

    folder = tmp_path_factory.mktemp("live-ui")
    with FeedServer() as feeds:
        settings = Settings(state_dir=folder, feeds_path=feeds.feeds_toml(folder / "feeds.toml"), public=True)
        engine = Engine.load(settings, source="live")
        port = _free_port()
        instance = uvicorn.Server(
            uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=instance.run, daemon=True)
        thread.start()
        for _ in range(100):
            if instance.started:
                break
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
        instance.should_exit = True
        thread.join(timeout=5)


def test_public_live_onboarding(browser, live_server):
    page = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(live_server)
    page.wait_for_selector("#reader-dialog[open]")
    assert page.locator("#dialog-title").inner_text() == "Make your feed"
    assert "Live edition" in page.locator("#edition").inner_text()
    page.locator('#dialog-categories .chip:has-text("Technology")').click()
    page.locator("#dialog-submit").click()  # no name needed
    page.wait_for_selector(".story")
    assert page.locator(".story").first.locator(".headline").inner_text().startswith("Battery start-up")
    assert "ranked for You" in page.locator("#edition").inner_text()

    # Another browser starts from scratch and cannot see this reader.
    other = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True).new_page()
    other.goto(live_server)
    other.wait_for_selector("#reader-dialog[open]")
    assert other.locator("#reader-select option").count() == 0
    assert not errors
