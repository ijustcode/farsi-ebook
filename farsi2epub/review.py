"""Interactive human-review workflow for correcting LLM transcriptions.

Serves a small local web app (stdlib http.server + jinja2, no external
network resources) that lets a human read each flagged page's source image
next to its transcribed Markdown, edit it if needed, and mark it
accepted/edited before EPUB assembly.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from jinja2 import Environment

from .workspace import PROJECT_ROOT, Workspace

DEFAULT_PORT = 8765
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"


# ---------------------------------------------------------------------------
# sidecar helpers
# ---------------------------------------------------------------------------


def _read_sidecar(ws: Workspace, n: int) -> dict:
    path = ws.page_meta_path(n)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_sidecar(ws: Workspace, n: int, data: dict) -> None:
    """Write the sidecar JSON atomically (write to a temp file, then replace)."""
    path = ws.page_meta_path(n)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    tmp_path.replace(path)


def _image_path_for(ws: Workspace, n: int) -> Optional[Path]:
    hi = ws.page_hires_path(n)
    if hi.is_file():
        return hi
    std = ws.page_image_path(n)
    if std.is_file():
        return std
    return None


# ---------------------------------------------------------------------------
# page selection (review budget)
# ---------------------------------------------------------------------------


def _select_pages_for_review(ws: Workspace) -> tuple[list[int], list[int]]:
    """Return (surfaced, skipped) page numbers.

    `surfaced` are the pages to actually show in the review UI: those with
    needs_review == true in their sidecar, capped at
    ceil(total_transcribed_pages / 5), keeping the lowest quality_score first.
    `skipped` are needs_review pages that were cut off by the budget; their
    sidecar gets a "review_skipped": true note but needs_review stays true.
    """
    done = ws.pages_done()
    total = len(done)
    budget = math.ceil(total / 5) if total else 0

    flagged: list[tuple[float, int]] = []
    for n in done:
        sidecar = _read_sidecar(ws, n)
        if sidecar.get("needs_review"):
            flagged.append((sidecar.get("quality_score", 0.0), n))

    # Lowest quality_score first.
    flagged.sort(key=lambda t: (t[0], t[1]))

    surfaced = [n for _, n in flagged[:budget]]
    skipped = [n for _, n in flagged[budget:]]

    for n in skipped:
        sidecar = _read_sidecar(ws, n)
        if not sidecar.get("review_skipped"):
            sidecar["review_skipped"] = True
            _write_sidecar(ws, n, sidecar)

    return sorted(surfaced), sorted(skipped)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<title>farsi2epub review &mdash; {{ slug }}</title>
<style>
@font-face {
  font-family: "Vazirmatn";
  src: url("/font/vazirmatn.ttf") format("truetype");
  font-weight: normal;
  font-style: normal;
}
* { box-sizing: border-box; }
body {
  font-family: "Vazirmatn", Tahoma, sans-serif;
  background: #1b1c20;
  color: #e8e8ea;
  margin: 0;
  padding: 0;
}
header {
  position: sticky;
  top: 0;
  background: #26272c;
  border-bottom: 1px solid #3a3b42;
  padding: 0.9rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  z-index: 10;
  direction: ltr;
}
header .title { font-size: 1.1rem; font-weight: 600; }
header .progress { font-size: 0.95rem; color: #b7b9c2; }
header a.done {
  background: #3a6ff0;
  color: white;
  padding: 0.45rem 1rem;
  border-radius: 6px;
  text-decoration: none;
  font-size: 0.9rem;
  cursor: pointer;
  border: none;
}
main { padding: 1.5rem; max-width: 1400px; margin: 0 auto; }
.skipped-note {
  background: #3a2f1c;
  border: 1px solid #6b5522;
  color: #f0d9a0;
  border-radius: 8px;
  padding: 0.8rem 1.2rem;
  margin-bottom: 1.5rem;
  direction: rtl;
  font-size: 0.95rem;
}
.page-block {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 2rem;
  padding: 1.2rem;
  border: 1px solid #3a3b42;
  border-radius: 10px;
  background: #212227;
  transition: opacity 0.2s;
}
.page-block.done { opacity: 0.55; }
.page-block .col-img { flex: 0 0 48%; max-width: 48%; }
.page-block .col-img img { max-width: 100%; border-radius: 6px; border: 1px solid #3a3b42; }
.page-block .col-text { flex: 0 0 48%; max-width: 48%; display: flex; flex-direction: column; }
.meta-row {
  direction: rtl;
  font-size: 0.85rem;
  color: #b7b9c2;
  margin-bottom: 0.6rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1rem;
}
.meta-row span.pill {
  background: #2f3037;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  border: 1px solid #3a3b42;
}
.flags { color: #f0a0a0; }
textarea {
  flex: 1;
  min-height: 420px;
  font-family: "Vazirmatn", Tahoma, sans-serif;
  font-size: 1.05rem;
  line-height: 1.8;
  direction: rtl;
  padding: 0.8rem;
  border-radius: 6px;
  border: 1px solid #3a3b42;
  background: #17181b;
  color: #e8e8ea;
  resize: vertical;
}
.actions { margin-top: 0.7rem; display: flex; gap: 0.6rem; direction: ltr; }
button {
  font-family: inherit;
  font-size: 0.9rem;
  padding: 0.5rem 1.1rem;
  border-radius: 6px;
  border: 1px solid #3a3b42;
  cursor: pointer;
}
button.save { background: #3a6ff0; color: white; border-color: #3a6ff0; }
button.accept { background: #2fa35a; color: white; border-color: #2fa35a; }
button:disabled { opacity: 0.5; cursor: default; }
.status-note { font-size: 0.85rem; color: #8fce9f; direction: ltr; align-self: center; }
</style>
</head>
<body>
<header>
  <div class="title">Review &mdash; {{ slug }}</div>
  <div class="progress" id="progress">{{ reviewed_count }} of {{ total_count }} reviewed</div>
  <a class="done" href="#" id="done-link">Done</a>
</header>
<main>
  {% if skipped %}
  <div class="skipped-note">
    {{ skipped|length }} additional flagged page(s) were auto-accepted despite flags due to the review budget: {{ skipped|join(', ') }}
  </div>
  {% endif %}

  {% for p in pages %}
  <div class="page-block{% if p.reviewed %} done{% endif %}" id="block-{{ p.page }}" data-page="{{ p.page }}">
    <div class="col-img">
      {% if p.image_url %}
      <img src="{{ p.image_url }}" alt="page {{ p.page }}">
      {% else %}
      <div>(no image available for page {{ p.page }})</div>
      {% endif %}
    </div>
    <div class="col-text">
      <div class="meta-row">
        <span class="pill">page {{ p.page }}</span>
        <span class="pill">model: {{ p.model_used }}</span>
        <span class="pill">confidence: {{ "%.2f"|format(p.confidence) }}</span>
        <span class="pill">quality: {{ "%.2f"|format(p.quality_score) }}</span>
        {% if p.flags %}
        <span class="pill flags">flags: {{ p.flags|join(', ') }}</span>
        {% endif %}
      </div>
      <textarea dir="rtl" lang="fa" id="text-{{ p.page }}">{{ p.text }}</textarea>
      <div class="actions">
        <button class="save" onclick="savePage({{ p.page }})">Save</button>
        <button class="accept" onclick="acceptPage({{ p.page }})">Accept</button>
        <span class="status-note" id="status-{{ p.page }}"></span>
      </div>
    </div>
  </div>
  {% endfor %}
</main>
<script>
function updateProgress(delta) {
  var el = document.getElementById('progress');
  var parts = el.textContent.match(/(\\d+) of (\\d+)/);
  if (!parts) return;
  var current = parseInt(parts[1], 10) + delta;
  el.textContent = current + ' of ' + parts[2] + ' reviewed';
}

function markDone(page) {
  var block = document.getElementById('block-' + page);
  if (block && !block.classList.contains('done')) {
    block.classList.add('done');
    updateProgress(1);
  }
}

async function savePage(page) {
  var text = document.getElementById('text-' + page).value;
  var status = document.getElementById('status-' + page);
  status.textContent = 'saving...';
  try {
    var resp = await fetch('/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({page: page, text: text})
    });
    var data = await resp.json();
    if (data.ok) {
      status.textContent = 'saved';
      markDone(page);
    } else {
      status.textContent = 'error: ' + (data.error || 'unknown');
    }
  } catch (e) {
    status.textContent = 'error: ' + e;
  }
}

async function acceptPage(page) {
  var status = document.getElementById('status-' + page);
  status.textContent = 'accepting...';
  try {
    var resp = await fetch('/accept', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({page: page})
    });
    var data = await resp.json();
    if (data.ok) {
      status.textContent = 'accepted';
      markDone(page);
    } else {
      status.textContent = 'error: ' + (data.error || 'unknown');
    }
  } catch (e) {
    status.textContent = 'error: ' + e;
  }
}

document.getElementById('done-link').addEventListener('click', async function (ev) {
  ev.preventDefault();
  await fetch('/quit', {method: 'POST'});
  document.body.innerHTML = '<main><h2 style="font-family:sans-serif;color:#eee;padding:2rem;">Review server stopped. You may close this tab.</h2></main>';
});
</script>
</body>
</html>
"""

_env = Environment(autoescape=True)
_TEMPLATE = _env.from_string(_PAGE_TEMPLATE)


def _relpath_for_image(ws: Workspace, path: Path) -> str:
    rel = path.relative_to(ws.root)
    return "/media/" + str(rel).replace("\\", "/")


def _render_index(ws: Workspace, surfaced: list[int], skipped: list[int]) -> str:
    pages = []
    reviewed_count = 0
    for n in surfaced:
        sidecar = _read_sidecar(ws, n)
        md_path = ws.page_md_path(n)
        text = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
        img_path = _image_path_for(ws, n)
        reviewed = not sidecar.get("needs_review", True)
        if reviewed:
            reviewed_count += 1
        pages.append(
            {
                "page": n,
                "model_used": sidecar.get("model_used", "?"),
                "confidence": sidecar.get("confidence", 0.0) or 0.0,
                "quality_score": sidecar.get("quality_score", 0.0) or 0.0,
                "flags": sidecar.get("flags", []) or [],
                "text": text,
                "image_url": _relpath_for_image(ws, img_path) if img_path else None,
                "reviewed": reviewed,
            }
        )
    return _TEMPLATE.render(
        slug=ws.slug,
        pages=pages,
        skipped=skipped,
        reviewed_count=reviewed_count,
        total_count=len(surfaced),
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _ReviewState:
    def __init__(self, ws: Workspace, surfaced: list[int], skipped: list[int]):
        self.ws = ws
        self.surfaced = surfaced
        self.skipped = skipped
        self.lock = threading.Lock()
        self.edited: set[int] = set()
        self.accepted: set[int] = set()
        self.httpd: Optional[ThreadingHTTPServer] = None


def _make_handler(state: _ReviewState):
    ws = state.ws

    class Handler(BaseHTTPRequestHandler):
        server_version = "farsi2epub-review/1.0"

        def log_message(self, fmt, *args):  # silence default stderr logging
            pass

        # -- helpers ---------------------------------------------------

        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        # -- routes ------------------------------------------------------

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                html = _render_index(ws, state.surfaced, state.skipped)
                self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/font/vazirmatn.ttf":
                if FONT_PATH.is_file():
                    self._send_bytes(FONT_PATH.read_bytes(), "font/ttf")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "font not found")
                return

            if path.startswith("/media/"):
                rel = path[len("/media/"):]
                candidate = (ws.root / rel).resolve()
                try:
                    candidate.relative_to(ws.root.resolve())
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN, "forbidden")
                    return
                if candidate.is_file():
                    self._send_bytes(candidate.read_bytes(), "image/png")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return

            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/save":
                try:
                    payload = self._read_json_body()
                    n = int(payload["page"])
                    text = payload["text"]
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                _write_text_atomic(ws.page_md_path(n), text)
                sidecar = _read_sidecar(ws, n)
                sidecar["needs_review"] = False
                sidecar["reviewed"] = "edited"
                _write_sidecar(ws, n, sidecar)
                with state.lock:
                    state.edited.add(n)
                    state.accepted.discard(n)
                self._send_json({"ok": True})
                return

            if path == "/accept":
                try:
                    payload = self._read_json_body()
                    n = int(payload["page"])
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                sidecar = _read_sidecar(ws, n)
                sidecar["needs_review"] = False
                sidecar["reviewed"] = "accepted"
                _write_sidecar(ws, n, sidecar)
                with state.lock:
                    state.accepted.add(n)
                    state.edited.discard(n)
                self._send_json({"ok": True})
                return

            if path == "/quit":
                self._send_json({"ok": True})

                def _shutdown():
                    if state.httpd is not None:
                        state.httpd.shutdown()

                threading.Thread(target=_shutdown, daemon=True).start()
                return

            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    return Handler


def _find_free_port(preferred: int) -> int:
    port = preferred
    for _ in range(200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue
            return port
    raise RuntimeError("could not find a free port")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_review(ws: Workspace, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    surfaced, skipped = _select_pages_for_review(ws)

    if not surfaced:
        if skipped:
            # Shouldn't really happen (skipped is a subset cut from surfaced
            # selection), but guard anyway.
            print(f"Nothing surfaced for review; {len(skipped)} page(s) auto-accepted despite flags.")
        else:
            print("Nothing needs review. All transcribed pages look good.")
        return

    if skipped:
        print(
            f"Review budget reached: {len(skipped)} flagged page(s) auto-accepted "
            f"despite flags (not shown): {skipped}"
        )

    state = _ReviewState(ws, surfaced, skipped)
    handler_cls = _make_handler(state)

    free_port = _find_free_port(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", free_port), handler_cls)
    state.httpd = httpd

    url = f"http://127.0.0.1:{free_port}/"
    print(f"Review server running at {url}")
    print(f"Surfaced {len(surfaced)} page(s) for review: {surfaced}")
    print("Press Ctrl+C when finished (or click Done in the page).")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        server_thread.join()
    except KeyboardInterrupt:
        httpd.shutdown()
        server_thread.join()
    finally:
        httpd.server_close()

    edited = len(state.edited)
    accepted = len(state.accepted)
    remaining = 0
    for n in surfaced:
        sidecar = _read_sidecar(ws, n)
        if sidecar.get("needs_review"):
            remaining += 1
    remaining += len(skipped)

    print("")
    print(f"Summary: edited {edited}, accepted {accepted}, remaining flagged {remaining}.")
