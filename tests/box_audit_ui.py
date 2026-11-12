"""Single-page UI of the box-audit tool (served by tests/box_audit.py).

No external assets: inline CSS/JS, the Vazirmatn font is served by the tool
from assets/fonts. Every action is sent to the server, which saves it before
answering; the page only shows what the server confirmed.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Box audit</title>
<style>
@font-face { font-family: "Vazirmatn"; src: url("/font/vazirmatn.ttf") format("truetype"); }
:root {
  --bg: #1b1c20; --panel: #24262c; --line: #3a3d45; --fg: #e8e8ea; --muted: #9a9ca5;
  --ok: #3fb950; --fixed: #58a6ff; --warn: #f0a030; --err: #f85149; --edit: #c58af9;
  --wa: #34d3ff; --wb: #ffd23f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.4 -apple-system, "Segoe UI", sans-serif; }
header { display: flex; gap: 8px; align-items: center; padding: 6px 10px; background: var(--panel); border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.tab { padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px; cursor: pointer; white-space: nowrap; }
.tab.cur { border-color: var(--fg); background: #30333a; }
.tab .st { font-size: 11px; margin-left: 6px; padding: 1px 5px; border-radius: 4px; background: #3a3d45; }
.tab .st.audited, .tab .st.previously_audited { background: #1f4d2a; color: #b8f5c4; }
#srv { margin-left: auto; font-size: 12px; color: var(--muted); }
#srv.down { color: var(--err); font-weight: 600; }
#banner { display: none; padding: 10px 14px; font-size: 14px; white-space: pre-wrap; }
#banner.err { display: block; background: #5a1d1d; color: #ffd7d5; border-bottom: 2px solid var(--err); }
#banner.warn { display: block; background: #4d3a12; color: #ffe8b5; border-bottom: 2px solid var(--warn); }
#banner code { background: #0008; padding: 1px 5px; border-radius: 4px; }
main { display: flex; gap: 10px; padding: 10px; align-items: flex-start; }
#left { flex: 1; min-width: 0; }
#right { width: 300px; flex: none; }
#info { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 4px; }
#info .big { font-size: 18px; font-weight: 600; }
.chip { padding: 1px 7px; border-radius: 10px; font-size: 12px; border: 1px solid var(--line); }
.chip.draft { color: var(--muted); } .chip.editing { color: var(--edit); border-color: var(--edit); }
.chip.accepted { color: var(--ok); border-color: var(--ok); } .chip.fixed { color: var(--fixed); border-color: var(--fixed); }
.chip.unc { color: var(--warn); border-color: var(--warn); } .chip.flag { color: var(--err); border-color: var(--err); }
#linetext { font-family: "Vazirmatn", Tahoma, sans-serif; font-size: 22px; direction: rtl; text-align: right; padding: 4px 8px; background: #15161a; border-radius: 6px; margin-bottom: 6px; }
#linetext .tok { padding: 0 2px; border-radius: 3px; }
#linetext .tok.sel { background: #ffffff30; outline: 1px solid #fff8; }
#linetext .tok.nobox { color: var(--err); text-decoration: underline wavy; }
#reasons { color: var(--warn); font-size: 13px; margin-bottom: 4px; }
#countmsg { color: var(--err); font-weight: 600; margin-bottom: 4px; }
#flagmsg { color: #ffb4ad; margin-bottom: 4px; }
#viewwrap { overflow: auto; background: #000; border: 1px solid var(--line); border-radius: 6px; max-height: 62vh; }
#view { display: block; touch-action: none; user-select: none; }
#view text { font-family: "Vazirmatn", Tahoma, sans-serif; direction: rtl; pointer-events: none; paint-order: stroke; stroke: #000; }
#toolbar { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
button { background: #30333a; color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 13px; }
button:hover { border-color: var(--fg); }
button.primary { background: #1f4d2a; border-color: var(--ok); }
button.on { background: #4b3a6a; border-color: var(--edit); }
button:disabled { opacity: .45; cursor: not-allowed; }
kbd { font: 11px monospace; background: #0006; border: 1px solid #555; border-radius: 3px; padding: 0 4px; }
#modeline { margin-top: 6px; color: var(--muted); font-size: 13px; min-height: 18px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px; margin-bottom: 8px; }
#pagestate.done { border-color: var(--ok); color: #b8f5c4; }
#thumb { display: block; width: 100%; height: auto; background: #000; border-radius: 4px; cursor: pointer; }
#wordinfo { font-size: 13px; }
#wordinfo .t { font-family: "Vazirmatn", Tahoma, sans-serif; font-size: 18px; direction: rtl; }
#help { font-size: 12.5px; color: var(--muted); columns: 2; column-gap: 18px; }
#help div { break-inside: avoid; margin-bottom: 2px; }
.muted { color: var(--muted); }
</style>
</head>
<body>
<header><div id="tabs"></div><div id="srv">connecting…</div></header>
<div id="banner"></div>
<main>
<section id="left">
  <div id="info"></div>
  <div id="linetext"></div>
  <div id="reasons"></div><div id="countmsg"></div><div id="flagmsg"></div>
  <div id="viewwrap"><svg id="view" xmlns="http://www.w3.org/2000/svg"></svg></div>
  <div id="modeline"></div>
  <div id="toolbar">
    <button class="primary" id="b-accept" title="Accept this line as correct (after any fixes) and go to the next line to review">Accept line <kbd>Enter</kbd></button>
    <button id="b-prev">Prev line <kbd>K</kbd></button>
    <button id="b-next">Next line <kbd>J</kbd></button>
    <button id="b-undo">Undo <kbd>Z</kbd></button>
    <button id="b-split" title="Split the selected box at the mouse position">Split box <kbd>S</kbd></button>
    <button id="b-merge" title="Merge the selected box with the box to its left">Merge with left <kbd>M</kbd></button>
    <button id="b-draw" title="Draw a new rectangle for the selected word (or a new box if none is selected)">Redraw <kbd>D</kbd></button>
    <button id="b-del" title="Delete the selected box">Delete box <kbd>Del</kbd></button>
    <button id="b-line" title="Edit the line outline: drag its edges, drag inside to move the whole line">Line outline <kbd>L</kbd></button>
    <button id="b-rotl">Rotate <kbd>[</kbd></button>
    <button id="b-rotr">Rotate <kbd>]</kbd></button>
    <button id="b-reseg" title="Re-run the automatic word split inside the current line outline">Re-split words <kbd>R</kbd></button>
    <button id="b-push" title="Move this line's last word to the start of the next line">Last word → next line <kbd>E</kbd></button>
    <button id="b-pull" title="Move the next line's first word to the end of this line">First word of next line → here <kbd>B</kbd></button>
    <button id="b-flag" title="Record that the truth text of this line is wrong (note only)">Text is wrong… <kbd>T</kbd></button>
    <button id="b-zin">Zoom + <kbd>+</kbd></button>
    <button id="b-zout">Zoom − <kbd>-</kbd></button>
  </div>
</section>
<aside id="right">
  <div class="panel" id="pagestate"></div>
  <div class="panel"><label><input type="checkbox" id="uncfirst" checked> Visit uncertain lines first</label></div>
  <div class="panel"><svg id="thumb" xmlns="http://www.w3.org/2000/svg"></svg>
    <div class="muted" style="font-size:12px;margin-top:4px">Click a line to go there. Grey: not reviewed, orange: uncertain, green: accepted, blue: fixed, purple: being edited, red outline: text flagged.</div></div>
  <div class="panel" id="wordinfo"></div>
  <div class="panel" id="help">
    <div><kbd>Enter</kbd>/<kbd>A</kbd> accept line, next</div>
    <div><kbd>J</kbd>/<kbd>↓</kbd> next line, <kbd>K</kbd>/<kbd>↑</kbd> previous</div>
    <div><kbd>Tab</kbd>/<kbd>←</kbd> next word (to the left), <kbd>Shift+Tab</kbd>/<kbd>→</kbd> previous</div>
    <div>Drag a dotted divider: move the boundary between two words</div>
    <div>Click a box to select; drag it to move; drag its handles to resize</div>
    <div><kbd>S</kbd> split selected box at the mouse, <kbd>M</kbd> merge with left box</div>
    <div><kbd>D</kbd> redraw selected box (or add a box), <kbd>Del</kbd> delete box</div>
    <div><kbd>L</kbd> line outline mode, <kbd>[</kbd> <kbd>]</kbd> rotate line 0.1° (Shift: 0.5°)</div>
    <div><kbd>R</kbd> re-split words inside the line outline</div>
    <div><kbd>E</kbd> last word to next line, <kbd>B</kbd> bring next line's first word here</div>
    <div><kbd>T</kbd> flag the line's text as wrong (note)</div>
    <div><kbd>Z</kbd>/<kbd>Ctrl/⌘+Z</kbd> undo, <kbd>+</kbd>/<kbd>-</kbd> zoom, <kbd>Esc</kbd> cancel</div>
  </div>
</aside>
</main>
<script>
"use strict";
var PORT = "__PORT__";
var RESTART = "./venv/bin/python tests/box_audit.py serve --port " + PORT;
var S = {pages: [], key: null, doc: null, line: 0, sel: null, mode: "word", zoom: 1, uncFirst: true,
         drag: null, mouseU: null, scale: 1, vb: null, down: false};
var NS = "http://www.w3.org/2000/svg";
function $(id) { return document.getElementById(id); }
function el(tag, attrs, parent) {
  var e = document.createElementNS(NS, tag);
  for (var k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

// ---------------------------------------------------------------- server I/O
function banner(msg, kind) {
  var b = $("banner");
  if (!msg) { b.className = ""; b.innerHTML = ""; return; }
  b.className = kind || "err";
  b.innerHTML = msg;
}
function downMessage(lost) {
  return (lost ? "<b>Not saved.</b> Your last action was NOT saved: " : "<b>The audit server is not reachable.</b> ") +
    "the box-audit server is not running (this browser could not reach http://127.0.0.1:" + PORT + "). " +
    "Everything saved before this message is safe on disk.\nRestart it in a terminal, from the repository folder:  <code>" +
    esc(RESTART) + "</code>  then reload this page. Your place is kept.";
}
function setServer(ok) {
  S.down = !ok;
  var s = $("srv");
  s.className = ok ? "" : "down";
  s.textContent = ok ? "server connected · saved" : "server not reachable";
}
async function api(path, body) {
  var r, d = null;
  try {
    r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  } catch (e) {
    setServer(false);
    banner(downMessage(body !== undefined), "err");
    render();
    return null;
  }
  setServer(true);
  try { d = await r.json(); } catch (e) { d = null; }
  if (r.status === 409 && d && d.doc) {
    S.doc = d.doc; render();
    banner("This page was changed elsewhere (another tab?). The saved state was reloaded; repeat your last action if it is missing.", "warn");
    return null;
  }
  if (!r.ok || !d || !d.ok) {
    banner("<b>Not saved:</b> " + esc((d && d.error) || ("the server answered HTTP " + r.status)), "err");
    render();
    return null;
  }
  if ($("banner").className) banner("");
  return d;
}
// Actions run one after another, each against the revision the previous one
// produced (no self-inflicted conflicts when keys are pressed quickly).
var chain = Promise.resolve();
function serial(fn) { var p = chain.then(fn); chain = p.catch(function () {}); return p; }
function op(name, args, lineIdx) { return serial(function () { return doOp(name, args, lineIdx); }); }
function undo() { return serial(doUndo); }
async function doOp(name, args, lineIdx) {
  if (!S.doc) return null;
  var d = await api("/api/op", {key: S.key, line: lineIdx === undefined ? S.line : lineIdx, op: name, args: args || {}, rev: S.doc.revision});
  if (d) { S.doc = d.doc; updateSummary(d.summary); render(); }
  return d;
}
async function doUndo() {
  var d = await api("/api/undo", {key: S.key});
  if (!d) return;
  S.doc = d.doc; updateSummary(d.summary);
  if (d.undone) { S.line = S.doc.cursor ? S.doc.cursor.line : S.line; S.sel = null; flash("Undid: " + d.undone.replace(/_/g, " ")); }
  else flash("Nothing to undo on this page.");
  render();
}
setInterval(async function () {
  try { var r = await fetch("/api/ping"); setServer(r.ok); }
  catch (e) { setServer(false); banner(downMessage(false), "err"); }
}, 10000);
function flash(msg) { $("modeline").textContent = msg; }

// ---------------------------------------------------------------- pages
function updateSummary(sm) {
  if (!sm) return;
  for (var i = 0; i < S.pages.length; i++) if (S.pages[i].key === sm.key) S.pages[i] = sm;
  renderTabs();
}
function renderTabs() {
  var t = $("tabs"); t.innerHTML = "";
  S.pages.forEach(function (p) {
    var d = document.createElement("span");
    d.className = "tab" + (p.key === S.key ? " cur" : "");
    if (p.error) { d.textContent = p.key + " (error: " + p.error + ")"; t.appendChild(d); return; }
    var st = p.status === "audited" ? "audited ✓" : p.status === "previously_audited" ? "previously audited" : p.done + "/" + p.lines;
    d.innerHTML = esc(p.label) + '<span class="st ' + p.status + '">' + esc(st) + "</span>";
    d.onclick = function () { openPage(p.key); };
    t.appendChild(d);
  });
}
async function openPage(key, line) {
  var d = await api("/api/page?key=" + encodeURIComponent(key));
  if (!d) return;
  S.key = key; S.doc = d.doc; S.sel = null; S.mode = "word"; S.zoom = 1;
  S.line = line != null ? line : (S.doc.cursor ? S.doc.cursor.line : 0);
  if (S.line >= S.doc.lines.length) S.line = 0;
  if (line == null && !S.doc.revision && S.uncFirst) { var q = queue(); if (q.length) S.line = q[0]; }
  store("box_audit_key", key);
  try { history.replaceState(null, "", "#key=" + encodeURIComponent(key)); } catch (e) {}
  var img = new Image(); img.src = "/img?key=" + encodeURIComponent(key);
  renderTabs(); render();
}
async function init() {
  var d = await api("/api/pages");
  if (!d) return;
  S.pages = d.pages;
  var uf = store("box_audit_uncfirst"); if (uf === "0") { S.uncFirst = false; $("uncfirst").checked = false; }
  var key = null, m = /key=([^&]+)/.exec(location.hash);
  if (m) key = decodeURIComponent(m[1]);
  if (!key) key = store("box_audit_key");
  if (!key || !S.pages.some(function (p) { return p.key === key; })) {
    var open = S.pages.filter(function (p) { return !p.error && p.status !== "audited" && p.status !== "previously_audited"; });
    key = (open[0] || S.pages[0]).key;
  }
  openPage(key);
}

// ---------------------------------------------------------------- navigation
function isDone(l) { return l.state === "accepted" || l.state === "fixed"; }
function queue() {
  var L = S.doc.lines, idx = [];
  for (var i = 0; i < L.length; i++) if (!isDone(L[i])) idx.push(i);
  if (S.uncFirst) idx.sort(function (a, b) { return (L[b].uncertain ? 1 : 0) - (L[a].uncertain ? 1 : 0) || a - b; });
  return idx;
}
function gotoLine(i) {
  if (!S.doc) return;
  i = Math.max(0, Math.min(S.doc.lines.length - 1, i));
  S.line = i; S.sel = null; S.mode = S.mode === "line" ? "line" : "word";
  render();
  op("set_cursor", {line: i}, i);
}
async function accept() {
  var d = await op("accept_line", {});
  if (!d) return;
  var q = queue();
  if (q.length) gotoLine(q[0]);
  else flash("Every line on this page is reviewed.");
}

// ---------------------------------------------------------------- rendering
function curLine() { return S.doc ? S.doc.lines[S.line] : null; }
function render() {
  if (!S.doc) return;
  var line = curLine();
  renderInfo(line); renderView(line); renderThumb(); renderPageState(); renderWordInfo(line); renderButtons(line);
}
function renderInfo(line) {
  var n = S.doc.lines.length;
  var chips = '<span class="chip ' + line.state + '">' + line.state + "</span>";
  if (line.uncertain) chips += ' <span class="chip unc">uncertain</span>';
  if (line.text_flag) chips += ' <span class="chip flag">text flagged</span>';
  $("info").innerHTML = '<span class="big">Line ' + (S.line + 1) + " / " + n + "</span> " + chips +
    ' <span class="muted">' + esc(S.doc.label || (S.doc.slug + " p" + S.doc.page)) + "</span>";
  var html = "";
  line.tokens.forEach(function (t, i) {
    var cls = "tok" + (i === S.sel ? " sel" : "") + (i >= line.words.length ? " nobox" : "");
    html += '<span class="' + cls + '" data-i="' + i + '">' + esc(t) + "</span> ";
  });
  $("linetext").innerHTML = html;
  Array.prototype.forEach.call($("linetext").querySelectorAll(".tok"), function (s) {
    s.onclick = function () { var i = +s.getAttribute("data-i"); if (i < line.words.length) { S.sel = i; render(); } };
  });
  $("reasons").textContent = line.uncertain && line.reasons.length ? "Draft unsure: " + line.reasons.join("; ") : "";
  var nb = line.words.length, nt = line.tokens.length;
  $("countmsg").textContent = nb === nt ? "" : (nb < nt
    ? (nt - nb) + " box(es) missing: " + nb + " boxes for " + nt + " words. Split a box (S) or draw one (D with nothing selected)."
    : (nb - nt) + " box(es) too many: " + nb + " boxes for " + nt + " words. Merge (M) or delete (Del) a box.");
  $("flagmsg").textContent = line.text_flag ? "Text flagged as wrong: " + line.text_flag.note : "";
}
function viewGeom(line) {
  var b = line.box, lh = Math.max(8, b[3] - b[1]);
  var u0 = b[0] - 0.5 * lh, u1 = b[2] + 0.5 * lh, v0 = b[1] - 1.5 * lh, v1 = b[3] + 1.1 * lh;
  var cw = Math.max(400, $("viewwrap").clientWidth - 2);
  var scale = Math.min(cw * S.zoom / (u1 - u0), 150 * S.zoom / lh);
  return {u0: u0, v0: v0, w: u1 - u0, h: v1 - v0, lh: lh, scale: scale};
}
function renderView(line) {
  var svg = $("view"); svg.innerHTML = "";
  var g = viewGeom(line); S.vb = g; S.scale = g.scale;
  svg.setAttribute("viewBox", g.u0 + " " + g.v0 + " " + g.w + " " + g.h);
  svg.setAttribute("width", Math.round(g.w * g.scale)); svg.setAttribute("height", Math.round(g.h * g.scale));
  var fr = line.frame, im = S.doc.image;
  var gi = el("g", {transform: "rotate(" + (-fr.angle_deg) + ") translate(" + (-fr.cx) + " " + (-fr.cy) + ")"}, svg);
  el("image", {href: "/img?key=" + encodeURIComponent(S.key), x: 0, y: 0, width: im.width_px, height: im.height_px}, gi);
  el("rect", {x: g.u0, y: g.v0, width: g.w, height: g.h, fill: "transparent", "data-kind": "bg"}, svg);
  var px = 1 / g.scale;  // one screen pixel in frame units
  var lb = line.box;
  el("rect", {x: lb[0], y: lb[1], width: lb[2] - lb[0], height: lb[3] - lb[1], fill: "none",
              stroke: S.mode === "line" ? "#c58af9" : "#ffffff88", "stroke-width": (S.mode === "line" ? 2 : 1) * px,
              "stroke-dasharray": (6 * px) + " " + (4 * px), "data-kind": S.mode === "line" ? "linebody" : "bg"}, svg);
  var fs = Math.max(0.36 * g.lh, 13 * px);
  // neighbouring lines' boxes, faint, so a word moved across a break is seen arriving
  [S.line - 1, S.line + 1].forEach(function (j) {
    var nl = S.doc.lines[j]; if (!nl) return;
    nl.words.forEach(function (w) {
      var pts = w.quad.map(function (q) { var f = pageToFrame(fr, q[0] * im.width_px, q[1] * im.height_px); return f[0] + "," + f[1]; }).join(" ");
      el("polygon", {points: pts, fill: "none", stroke: "#8a5cf6", "stroke-opacity": 0.7, "stroke-width": 1.5 * px, "stroke-dasharray": (3 * px) + " " + (3 * px), "pointer-events": "none"}, svg);
    });
  });
  line.words.forEach(function (w, i) {
    var b = w.box, col = i % 2 ? "var(--wb)" : "var(--wa)";
    var sel = i === S.sel;
    var r = el("rect", {x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1],
                        fill: sel ? "#ffffff22" : "#ffffff05", stroke: w.uncertain && w.state === "draft" ? "var(--warn)" : col,
                        "stroke-width": (sel ? 3 : 2) * px, "data-kind": "word", "data-i": i}, svg);
    if (w.uncertain && w.state === "draft") r.setAttribute("stroke-dasharray", (5 * px) + " " + (3 * px));
    var t = w.text == null ? "∅ no word" : w.text + (w.uncertain && w.state === "draft" ? " ?" : "");
    var ty = b[1] - (i % 2 ? 1.05 : 0.2) * fs - 2 * px;
    var tx = el("text", {x: (b[0] + b[2]) / 2, y: ty, "font-size": fs, "text-anchor": "middle",
                         fill: w.text == null ? "var(--err)" : col, "stroke-width": 3 * px}, svg);
    tx.textContent = t;
  });
  // boundary dividers between neighbouring words (word i on the right, i+1 on its left)
  for (var i = 0; i + 1 < line.words.length; i++) {
    var a = line.words[i].box, b2 = line.words[i + 1].box;
    if (a[0] < b2[2] - 0.5 * (b2[2] - b2[0])) continue;  // not neighbours in this order
    var u = (a[0] + b2[2]) / 2;
    el("line", {x1: u, x2: u, y1: lb[1], y2: lb[3], stroke: "#ffffffaa", "stroke-width": 1 * px,
                "stroke-dasharray": (2 * px) + " " + (3 * px), "pointer-events": "none"}, svg);
    // grab area: the white gap itself plus 3 screen pixels on each side, so
    // it never covers a narrow word box
    var gw = Math.max(0, a[0] - b2[2]) + 6 * px;
    el("rect", {x: u - gw / 2, y: lb[1], width: gw, height: lb[3] - lb[1], fill: "transparent",
                style: "cursor: col-resize", "data-kind": "boundary", "data-i": i}, svg);
  }
  if (S.sel != null && line.words[S.sel] && S.mode === "word") handles(svg, line.words[S.sel].box, "handle", px);
  if (S.mode === "line") handles(svg, lb, "linehandle", px);
  var mode = {word: "Click a box to select it; drag dotted dividers to move a word boundary.",
              draw: "Draw mode: drag a rectangle " + (S.sel != null ? "for the selected word" : "to add a box") + " (Esc cancels).",
              line: "Line outline mode: drag the outline's edges, or drag inside it to move the whole line. L or Esc to leave."}[S.mode];
  if (!S.drag) flash(mode);
  if (S.zoom > 1 && S.sel != null && line.words[S.sel]) {
    var bb = line.words[S.sel].box, wrap = $("viewwrap");
    var left = (bb[0] - g.u0) * g.scale, right = (bb[2] - g.u0) * g.scale;
    if (left < wrap.scrollLeft || right > wrap.scrollLeft + wrap.clientWidth) wrap.scrollLeft = (left + right) / 2 - wrap.clientWidth / 2;
  }
}
function pageToFrame(fr, x, y) {
  var a = fr.angle_deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), dx = x - fr.cx, dy = y - fr.cy;
  return [dx * c + dy * s, -dx * s + dy * c];
}
async function moveWord(kind) {
  var L = curLine(), nxt = S.doc.lines[S.line + 1];
  if (!nxt) return;
  var word = kind === "push_word" ? L.tokens[L.tokens.length - 1] : nxt.tokens[0];
  var d = await op(kind, {});
  if (d) flash((kind === "push_word" ? "Moved “" + word + "” to the start of line " + (S.line + 2) + ", which was re-split (check it with J)."
                                     : "Moved “" + word + "” to the end of this line, which was re-split; line " + (S.line + 2) + " lost its first box.") +
               " Both lines need Enter again. Z undoes both.");
}
function handles(svg, b, kind, px) {
  var s = 9 * px, cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  var pts = {tl: [b[0], b[1]], tr: [b[2], b[1]], bl: [b[0], b[3]], br: [b[2], b[3]], l: [b[0], cy], r: [b[2], cy], t: [cx, b[1]], b: [cx, b[3]]};
  var cur = {tl: "nwse", br: "nwse", tr: "nesw", bl: "nesw", l: "ew", r: "ew", t: "ns", b: "ns"};
  for (var k in pts) el("rect", {x: pts[k][0] - s / 2, y: pts[k][1] - s / 2, width: s, height: s, fill: "#fff", stroke: "#000",
                                 "stroke-width": px, "data-kind": kind, "data-edge": k, style: "cursor:" + cur[k] + "-resize"}, svg);
}
function lineColor(l) {
  if (l.state === "accepted") return "#3fb950";
  if (l.state === "fixed") return "#58a6ff";
  if (l.state === "editing") return "#c58af9";
  return l.uncertain ? "#f0a030" : "#9a9ca5";
}
function renderThumb() {
  var svg = $("thumb"), im = S.doc.image, W = im.width_px, H = im.height_px;
  svg.innerHTML = "";
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  el("image", {href: "/img?key=" + encodeURIComponent(S.key), x: 0, y: 0, width: W, height: H, opacity: 0.55}, svg);
  S.doc.lines.forEach(function (l, i) {
    var pts = l.quad.map(function (p) { return (p[0] * W) + "," + (p[1] * H); }).join(" ");
    var p = el("polygon", {points: pts, fill: lineColor(l), "fill-opacity": i === S.line ? 0.55 : 0.28,
                           stroke: l.text_flag ? "#f85149" : (i === S.line ? "#fff" : lineColor(l)),
                           "stroke-width": i === S.line || l.text_flag ? W / 160 : W / 500}, svg);
    p.style.cursor = "pointer";
    p.onclick = function () { gotoLine(i); };
  });
  (S.doc.regions || []).forEach(function (r) {
    el("rect", {x: r.rect[0] * W, y: r.rect[1] * H, width: (r.rect[2] - r.rect[0]) * W, height: (r.rect[3] - r.rect[1]) * H,
                fill: "none", stroke: "#c58af9", "stroke-width": W / 400, "stroke-dasharray": W / 100}, svg);
  });
}
function renderPageState() {
  var L = S.doc.lines, done = L.filter(isDone).length, unc = L.filter(function (l) { return l.uncertain && !isDone(l); }).length;
  var words = L.reduce(function (a, l) { return a + l.words.length; }, 0);
  var ps = $("pagestate"), st = S.doc.status, html;
  if (st === "audited") html = "<b>Page audited ✓</b><br>Every line was explicitly accepted. Audited " + esc(S.doc.audited_at || "") + ".";
  else if (st === "previously_audited") html = "<b>Previously audited</b> (" + esc((S.doc.previous_audit || {}).auditor || "") + ", " +
      esc(((S.doc.previous_audit || {}).audited_at || "").slice(0, 10)) + ").<br>Spot-check it. An edit re-opens that line until you accept it again.";
  else html = "<b>" + done + " / " + L.length + "</b> lines reviewed · " + unc + " uncertain still open · " + words + " boxes";
  html += '<div class="muted" style="font-size:12px;margin-top:4px">Text source: ' + esc(S.doc.text_source.kind.replace(/_/g, " ")) + "</div>";
  ps.innerHTML = html;
  ps.className = "panel" + (st === "audited" || st === "previously_audited" ? " done" : "");
}
function renderWordInfo(line) {
  var w = S.sel != null ? line.words[S.sel] : null, h;
  if (!w) h = '<span class="muted">No box selected. ' + line.words.length + " boxes, " + line.tokens.length + " words on this line.</span>";
  else h = 'Box ' + (S.sel + 1) + ' of ' + line.words.length + ' · <span class="t">' + esc(w.text == null ? "∅ (no word)" : w.text) + "</span><br>state: " + w.state +
    (w.uncertain ? '<br><span style="color:var(--warn)">draft unsure: ' + esc((w.reasons || []).join("; ")) + "</span>" : "");
  $("wordinfo").innerHTML = h;
}
function renderButtons(line) {
  var hasSel = S.sel != null && line.words[S.sel];
  $("b-accept").disabled = !(line.words.length === line.tokens.length || line.text_flag);
  $("b-split").disabled = !hasSel; $("b-del").disabled = !hasSel;
  $("b-merge").disabled = !(hasSel && S.sel + 1 < line.words.length);
  $("b-draw").className = S.mode === "draw" ? "on" : ""; $("b-line").className = S.mode === "line" ? "on" : "";
  $("b-push").disabled = S.line + 1 >= S.doc.lines.length; $("b-pull").disabled = S.line + 1 >= S.doc.lines.length;
}

// ---------------------------------------------------------------- pointer
function framePt(ev) {
  var svg = $("view"), p = svg.createSVGPoint();
  p.x = ev.clientX; p.y = ev.clientY;
  var q = p.matrixTransform(svg.getScreenCTM().inverse());
  return {u: q.x, v: q.y};
}
window.addEventListener("pointermove", function (ev) {
  if (!S.doc || !$("view").getScreenCTM()) return;
  var p = framePt(ev); S.mouseU = p.u;
  var d = S.drag; if (!d) return;
  d.moved = d.moved || Math.abs(p.u - d.p0.u) * S.scale > 3 || Math.abs(p.v - d.p0.v) * S.scale > 3;
  d.p = p;
  preview();
});
$("view").addEventListener("pointerdown", function (ev) {
  if (!S.doc || ev.button !== 0) return;
  if (S.drag && S.drag.moved) finishDrag();   // a drag whose release was lost
  var t = ev.target, kind = t.getAttribute("data-kind"), p = framePt(ev), line = curLine();
  try { $("view").setPointerCapture(ev.pointerId); } catch (e) {}
  if (S.mode === "draw") { S.drag = {kind: "draw", p0: p, p: p}; return; }
  if (kind === "boundary") { S.drag = {kind: "boundary", i: +t.getAttribute("data-i"), p0: p, p: p}; return; }
  if (kind === "handle") { S.drag = {kind: "resize", i: S.sel, edge: t.getAttribute("data-edge"), box: line.words[S.sel].box.slice(), p0: p, p: p}; return; }
  if (kind === "linehandle") { S.drag = {kind: "lineresize", edge: t.getAttribute("data-edge"), box: line.box.slice(), p0: p, p: p}; return; }
  if (S.mode === "line" && (kind === "linebody" || kind === "word")) { S.drag = {kind: "linemove", p0: p, p: p}; return; }
  if (kind === "word") {
    var i = +t.getAttribute("data-i");
    if (S.sel !== i) { S.sel = i; render(); }
    S.drag = {kind: "move", i: i, box: line.words[i].box.slice(), p0: p, p: p};
    return;
  }
  if (S.sel != null) { S.sel = null; render(); }
});
function resized(box, edge, du, dv) {
  var b = box.slice();
  if (edge.indexOf("l") >= 0) b[0] += du;
  if (edge.indexOf("r") >= 0) b[2] += du;
  if (edge.indexOf("t") >= 0) b[1] += dv;
  if (edge.indexOf("b") >= 0) b[3] += dv;
  return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
}
function dragResult(d) {
  d = d || S.drag;
  var du = d.p.u - d.p0.u, dv = d.p.v - d.p0.v;
  if (d.kind === "move") return [d.box[0] + du, d.box[1] + dv, d.box[2] + du, d.box[3] + dv];
  if (d.kind === "resize") return resized(d.box, d.edge, du, dv);
  if (d.kind === "lineresize") return resized(d.box, d.edge, du, dv);
  if (d.kind === "draw") return [Math.min(d.p0.u, d.p.u), Math.min(d.p0.v, d.p.v), Math.max(d.p0.u, d.p.u), Math.max(d.p0.v, d.p.v)];
  return null;
}
function preview() {
  var d = S.drag, svg = $("view"), old = document.getElementById("pv");
  if (old) old.parentNode.removeChild(old);
  if (!d.moved) return;
  var px = 1 / S.scale, line = curLine();
  if (d.kind === "boundary") {
    el("line", {id: "pv", x1: d.p.u, x2: d.p.u, y1: line.box[1], y2: line.box[3], stroke: "#ff4fd8", "stroke-width": 2 * px}, svg);
    return;
  }
  if (d.kind === "linemove") {
    var b = line.box;
    el("rect", {id: "pv", x: b[0] + d.p.u - d.p0.u, y: b[1] + d.p.v - d.p0.v, width: b[2] - b[0], height: b[3] - b[1],
                fill: "none", stroke: "#ff4fd8", "stroke-width": 2 * px}, svg);
    return;
  }
  var r = dragResult();
  el("rect", {id: "pv", x: r[0], y: r[1], width: r[2] - r[0], height: r[3] - r[1], fill: "#ff4fd822", stroke: "#ff4fd8", "stroke-width": 2 * px}, svg);
}
window.addEventListener("pointerup", finishDrag);
window.addEventListener("pointercancel", function () { finishDrag(null); });
async function finishDrag(ev) {
  var d = S.drag; S.drag = null;
  if (!d) return;
  if (ev && ev.clientX != null && $("view").getScreenCTM()) {   // the release point counts even without moves
    var p = framePt(ev);
    d.moved = d.moved || Math.abs(p.u - d.p0.u) * S.scale > 3 || Math.abs(p.v - d.p0.v) * S.scale > 3;
    if (d.moved) d.p = p;
  }
  if (!d.moved) { if (d.kind === "draw") return; render(); return; }
  var line = curLine();
  if (d.kind === "boundary") return op("set_boundary", {word: d.i, u: d.p.u});
  if (d.kind === "move" || d.kind === "resize") return op("set_word_box", {word: d.i, box: dragResult(d)});
  if (d.kind === "lineresize") return op("set_line_box", {box: dragResult(d)});
  if (d.kind === "linemove") return op("move_line", {du: d.p.u - d.p0.u, dv: d.p.v - d.p0.v});
  if (d.kind === "draw") {
    var r = dragResult(d); S.mode = "word";
    if (S.sel != null && line.words[S.sel]) return op("set_word_box", {word: S.sel, box: r});
    var c = (r[0] + r[2]) / 2, idx = 0;
    line.words.forEach(function (w) { if ((w.box[0] + w.box[2]) / 2 > c) idx++; });
    var res = await op("add_box", {box: r, index: idx});
    if (res) { S.sel = idx; render(); }
  }
}

// ---------------------------------------------------------------- actions
function doSplit() {
  var line = curLine(); if (S.sel == null || !line.words[S.sel]) return;
  var b = line.words[S.sel].box, u = S.mouseU;
  if (u == null || u <= b[0] + 2 || u >= b[2] - 2) u = (b[0] + b[2]) / 2;
  op("split", {word: S.sel, u: u});
}
function doMerge() { var line = curLine(); if (S.sel != null && S.sel + 1 < line.words.length) op("merge", {word: S.sel}); }
function doDelete() { var line = curLine(); if (S.sel != null && line.words[S.sel]) { var i = S.sel; S.sel = null; op("delete_box", {word: i}); } }
function doFlag() {
  var line = curLine();
  if (line.text_flag) { if (confirm("Remove the text flag on this line?\n\nNote: " + line.text_flag.note)) op("unflag_text", {}); return; }
  var note = prompt("What is wrong with the truth text of this line? (Only this note is saved; the text is not edited here.)", "");
  if (note && note.trim()) op("flag_text", {note: note.trim()});
}
function setMode(m) { S.mode = S.mode === m ? "word" : m; render(); }
function zoom(f) { S.zoom = Math.max(1, Math.min(5, S.zoom * f)); render(); }
$("b-accept").onclick = accept;
$("b-prev").onclick = function () { gotoLine(S.line - 1); };
$("b-next").onclick = function () { gotoLine(S.line + 1); };
$("b-undo").onclick = undo;
$("b-split").onclick = doSplit;
$("b-merge").onclick = doMerge;
$("b-del").onclick = doDelete;
$("b-draw").onclick = function () { setMode("draw"); };
$("b-line").onclick = function () { setMode("line"); };
$("b-rotl").onclick = function (e) { op("rotate_line", {delta_deg: e.shiftKey ? -0.5 : -0.1}); };
$("b-rotr").onclick = function (e) { op("rotate_line", {delta_deg: e.shiftKey ? 0.5 : 0.1}); };
$("b-reseg").onclick = async function () {
  var before = curLine().box.slice();
  var d = await op("resegment", {});
  var after = curLine().box;
  if (d) flash("Re-split the words inside the line outline" + ((after[3] - after[1]) < (before[3] - before[1]) - 1 ? " (outline first shrunk back to this line's own ink)" : "") + ". Z undoes.");
};
$("b-push").onclick = function () { moveWord("push_word"); };
$("b-pull").onclick = function () { moveWord("pull_word"); };
$("b-flag").onclick = doFlag;
$("b-zin").onclick = function () { zoom(1.5); };
$("b-zout").onclick = function () { zoom(1 / 1.5); };
$("uncfirst").onchange = function () { S.uncFirst = this.checked; store("box_audit_uncfirst", this.checked ? "1" : "0"); };
window.addEventListener("resize", function () { if (S.doc) render(); });
document.addEventListener("keydown", function (ev) {
  if (!S.doc || ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;
  var k = ev.key, line = curLine();
  if ((ev.metaKey || ev.ctrlKey) && (k === "z" || k === "Z")) { ev.preventDefault(); undo(); return; }
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  var handled = true;
  switch (k) {
    case "Enter": case "a": case "A": accept(); break;
    case "j": case "J": case "ArrowDown": gotoLine(S.line + 1); break;
    case "k": case "K": case "ArrowUp": gotoLine(S.line - 1); break;
    case "n": case "N": var q = queue(); if (q.length) gotoLine(q[0]); break;
    case "Tab": case "ArrowLeft": case "ArrowRight":
      var back = (k === "Tab" && ev.shiftKey) || k === "ArrowRight", n = line.words.length;
      if (n) { S.sel = S.sel == null ? (back ? n - 1 : 0) : (S.sel + (back ? -1 : 1) + n) % n; render(); }
      break;
    case "z": case "Z": case "u": case "U": undo(); break;
    case "s": case "S": doSplit(); break;
    case "m": case "M": doMerge(); break;
    case "Delete": case "Backspace": doDelete(); break;
    case "d": case "D": setMode("draw"); break;
    case "l": case "L": setMode("line"); break;
    case "[": op("rotate_line", {delta_deg: -0.1}); break;
    case "{": op("rotate_line", {delta_deg: -0.5}); break;
    case "]": op("rotate_line", {delta_deg: 0.1}); break;
    case "}": op("rotate_line", {delta_deg: 0.5}); break;
    case "r": case "R": $("b-reseg").onclick(); break;
    case "e": case "E": moveWord("push_word"); break;
    case "b": case "B": moveWord("pull_word"); break;
    case "t": case "T": doFlag(); break;
    case "+": case "=": zoom(1.5); break;
    case "-": case "_": zoom(1 / 1.5); break;
    case "Escape": S.mode = "word"; S.drag = null; S.sel = null; render(); break;
    default: handled = false;
  }
  if (handled) ev.preventDefault();
});
init();
</script>
</body>
</html>
"""
