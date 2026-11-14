"""Single-page UI for tests/text_audit.py (no external assets). ``PAGE_HTML`` has two
placeholders replaced by the server: __PORT__ and __CMD__ (the restart command).

Layout preview: ``/?w=1800`` (optionally ``&h=900``) lays the page out as if the window were that many CSS
pixels wide (and high) and scales the result to the real window, so wide layouts can be inspected in a small
window. It is a scaled picture: text is smaller than it would be, and real-size sharpness and the way the
browser wraps text at its true size are not shown."""

PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Text audit</title>
<style>
@font-face{font-family:VzLocal;src:url(/font.ttf);}
:root{--bg:#f7f6f2;--pn:#fff;--fg:#1d1d1b;--mut:#6b6b66;--ln:#d9d7cf;--ac:#1f5fbf;--acbg:#e3edfb;--ok:#18794e;--okbg:#e1f3e9;
--wn:#9a5b00;--wnbg:#fff1d6;--er:#b3261e;--erbg:#fde7e5;--hl:#ffe08a;--vh:100vh}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--pn:#242422;--fg:#ecece8;--mut:#a0a09a;--ln:#3d3d3a;--ac:#8ab4ff;--acbg:#23324d;--ok:#6fd1a0;--okbg:#1d3a2c;--wn:#f0b45a;--wnbg:#41320f;--er:#ff9a92;--erbg:#4a1f1c;--hl:#7a5d00}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,sans-serif}
#viewport{position:relative}
.preview #viewport{width:100vw;height:100vh;overflow:hidden}
#app{position:relative;transform-origin:0 0;background:var(--bg)}
button{font:inherit;color:inherit;background:var(--pn);border:1px solid var(--ln);border-radius:8px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--ac)}button:disabled{opacity:.45;cursor:default}
button.pri{background:var(--ac);color:#fff;border-color:var(--ac)}
button.sm{font-size:12px;padding:3px 8px;border-radius:6px}
.fa{font-family:VzLocal,"Geeza Pro","SF Arabic","Noto Naskh Arabic",Tahoma,serif}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.mut{color:var(--mut)}
#top{height:42px;display:flex;gap:8px;align-items:center;padding:0 12px;background:var(--pn);border-bottom:1px solid var(--ln);overflow:hidden;white-space:nowrap}
.tab{padding:3px 10px}.tab.on{background:var(--acbg);border-color:var(--ac);font-weight:600}.tab small{color:var(--mut);margin-left:6px;font-size:11px}
#status{margin-left:auto;color:var(--mut);font-size:13px}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-weight:600;font-size:12px}
.badge.fin{background:var(--okbg);color:var(--ok)}.badge.nf{background:var(--wnbg);color:var(--wn)}
#cols{display:grid;gap:10px;padding:8px 10px;height:calc(var(--vh) - 42px)}
.L-wide #cols{grid-template-columns:minmax(560px,1.25fr) minmax(0,1fr) minmax(300px,.8fr)}
.L-mid #cols{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);grid-template-rows:minmax(0,1fr) minmax(0,1fr)}
.L-mid #core{grid-row:1/3}.L-mid #pagecol{grid-column:2;grid-row:1}.L-mid #listcol{grid-column:2;grid-row:2}
.L-narrow #app{height:auto!important}.L-narrow #cols{display:block;height:auto}
.L-narrow #core{margin-bottom:10px}.L-narrow #pgwrap{height:55vh}.L-narrow #list{max-height:50vh}
.col{min-width:0;min-height:0;display:flex;flex-direction:column;gap:8px}
.card{background:var(--pn);border:1px solid var(--ln);border-radius:10px;padding:8px 10px;min-width:0}
#core .card{display:flex;flex-direction:column;min-height:0}
#cropcard{flex:none}
#editcard{flex:1;min-height:0}
#cropbar{margin-bottom:5px}
#stripbox{background:#fff;border:1px solid var(--ln);border-radius:6px;overflow:hidden}
.srow{position:relative;overflow:hidden;line-height:0}
.srow img{position:absolute;top:0;max-width:none}
.srow .dim{position:absolute;left:0;right:0;background:rgba(40,40,40,.5);pointer-events:none}
.srow .tag{position:absolute;right:3px;top:2px;font:700 10px system-ui;color:#fff;background:#555;border-radius:4px;padding:0 4px;line-height:14px;opacity:.8;z-index:3}
.srow+.srow{border-top:3px solid #888}
.ptr{position:absolute;top:0;width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;border-top:11px solid #1f5fbf;margin-left:-7px;opacity:.55;z-index:2}
.ptr.act{opacity:1;border-top-color:#d9480f;border-left-width:10px;border-right-width:10px;border-top-width:16px;margin-left:-10px}
#croppanel{margin-top:6px;padding:8px;border:1px solid var(--ac);border-radius:8px;background:var(--acbg)}
#croppanel .tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
#croppanel .t{border:2px solid var(--ln);border-radius:8px;background:var(--pn);padding:3px;cursor:pointer;font-size:11px;overflow:hidden}
#croppanel .t.cur{border-color:var(--ok)}#croppanel .t.sel{border-color:var(--ac)}
#croppanel .t img{display:block;width:100%;height:46px;object-fit:cover;object-position:right center;background:#fff}
#editor{flex:1;min-height:0;overflow:auto}
.ed{direction:rtl;font-size:28px;line-height:2.15;padding:4px;white-space:pre-wrap;overflow-wrap:anywhere}
.ed .w{cursor:pointer;border-radius:4px;padding:0 1px}.ed .w:hover{background:var(--hl)}
.mk{display:inline-block;vertical-align:middle;margin:2px 4px;padding:3px 5px;border-radius:10px;border:2px solid var(--wn);background:var(--wnbg);direction:rtl;white-space:normal;line-height:1.5}
.mk.done{border-color:var(--ok);background:var(--okbg)}.mk.act{box-shadow:0 0 0 3px var(--ac)}
.mk .alts{display:flex;flex-wrap:wrap;gap:5px}
button.alt{font-size:22px;line-height:1.65;padding:1px 9px;text-align:right;direction:rtl;max-width:100%;white-space:normal}
button.alt .k{display:inline-block;direction:ltr;font:700 12px system-ui;background:var(--ac);color:#fff;border-radius:999px;min-width:18px;height:18px;line-height:18px;text-align:center;margin-left:7px;vertical-align:middle}
button.alt.on{background:var(--ok);color:#fff;border-color:var(--ok)}
button.alt .plain{font:600 14px system-ui;direction:ltr;unicode-bidi:isolate}
button.alt.gap{border-style:dashed}
i.sp{font-style:normal;color:#d9480f;font-size:.62em;margin:0 1px;font-family:system-ui}
i.zw{font:700 .38em system-ui;color:#fff;background:#d9480f;border-radius:4px;padding:1px 3px;margin:0 1px;direction:ltr;unicode-bidi:isolate;vertical-align:middle}
.mkact{margin-top:4px;display:flex;gap:5px;flex-wrap:wrap;direction:ltr}
.mknote{direction:ltr;font:11px/1.3 system-ui;color:var(--wn);margin-top:3px;max-width:520px;text-align:left}
.chosen{font-size:24px;direction:rtl;padding:0 8px;background:transparent;border:none}
.brk{border:2px solid var(--wn);background:var(--wnbg);border-radius:10px;padding:6px 10px;margin-bottom:6px;font-size:13px}
.brk .other{direction:rtl;font-size:15px;color:var(--mut);margin:3px 0;line-height:1.4;max-height:4.4em;overflow:hidden}
#tb{flex:none;border:2px solid var(--ac);border-radius:10px;padding:8px;margin-top:6px;background:var(--acbg)}
#tbin{width:100%;font-size:26px;direction:rtl;padding:4px 10px;border-radius:8px;border:1px solid var(--ln);background:var(--pn);color:var(--fg)}
#nav{flex:none;margin-top:6px;display:flex;flex-direction:column;gap:5px}
#list{flex:1;min-height:0;overflow:auto}
.li{display:flex;gap:8px;align-items:center;padding:2px 6px;border-radius:6px;cursor:pointer}
.li:hover{background:var(--acbg)}.li.cur{background:var(--hl);color:#111}
.li .n{width:26px;color:var(--mut);text-align:right;flex:none;font-size:12px}
.li .st{width:68px;flex:none;font-size:11px}
.li .tx{flex:1;min-width:0;direction:rtl;text-align:right;font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.li .tx mark{background:var(--wnbg);color:var(--wn);border-radius:4px}
.st.ok{color:var(--ok)}.st.wn{color:var(--wn)}
.unm{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center;margin:3px 0;padding:3px 6px;border:1px dashed var(--wn);border-radius:8px;background:var(--wnbg);font-size:12px;color:var(--wn)}
.unm.f{box-shadow:0 0 0 2px var(--ac)}
.unm img{width:90px;height:26px;object-fit:cover;object-position:right center;background:#fff;border-radius:3px;flex:none;cursor:pointer}
.unm .m{flex:1 1 120px;min-width:0}
#pgwrap{flex:1;min-height:0;overflow:auto;border:1px solid var(--ln);border-radius:6px;background:#ddd}
#pgbox{position:relative;line-height:0}#pgimg{width:100%;display:block}#pgsvg{position:absolute;left:0;top:0;width:100%;height:100%}
#toasts{position:absolute;left:50%;transform:translateX(-50%);top:48px;z-index:60;display:flex;flex-direction:column;gap:6px;align-items:center;pointer-events:none;width:max-content;max-width:min(620px,92%)}
.toast{pointer-events:auto;padding:8px 14px;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.3);font-size:14px;display:flex;gap:10px;align-items:center;background:var(--pn);border:2px solid var(--ln)}
.toast.err{background:var(--erbg);color:var(--er);border-color:var(--er)}.toast.warn{background:var(--wnbg);color:var(--wn);border-color:var(--wn)}.toast.info{background:var(--okbg);color:var(--ok);border-color:var(--ok)}
#down{display:none;position:absolute;left:10px;right:10px;top:46px;z-index:70;padding:12px 14px;border-radius:10px;background:var(--erbg);color:var(--er);font-weight:500;box-shadow:0 4px 18px rgba(0,0,0,.3)}
#down code{display:block;margin-top:6px;padding:6px 8px;background:var(--pn);color:var(--fg);border-radius:6px;user-select:all;word-break:break-all;font-weight:400}
.modal{position:absolute;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:50;display:none;align-items:flex-start;justify-content:center;padding:30px 10px;overflow:auto}
.modal.on{display:flex}.modal .box{background:var(--pn);border-radius:12px;max-width:820px;width:100%;padding:16px 20px}
.pick{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}
.pick .c{border:2px solid var(--ln);border-radius:8px;padding:4px;cursor:pointer;background:var(--pn);font-size:12px}
.pick .c.on{border-color:var(--ac)}.pick .c img{width:100%;display:block;background:#fff}
kbd{background:var(--bg);border:1px solid var(--ln);border-bottom-width:2px;border-radius:4px;padding:0 5px;font:12px monospace}
table.k td{padding:2px 12px 2px 0;vertical-align:top}
</style></head><body>
<div id="viewport"><div id="app">
<div id="top">
 <b>Text audit</b><span id="tabs" class="row" style="flex-wrap:nowrap"></span>
 <button id="undo" class="sm" title="Undo the last action (Z or Ctrl/Cmd+Z)">Undo (Z)</button>
 <button id="helpb" class="sm">Help (?)</button>
 <span id="status"></span>
</div>
<div id="cols">
 <section id="core" class="col">
  <div class="card" id="cropcard">
   <div id="cropbar" class="row"><b id="lineinfo"></b><span id="lineflags"></span><span id="cropinfo" class="mut" style="font-size:12px"></span>
    <span style="margin-left:auto" class="row"><button id="wrongstrip" class="sm">Wrong strip?</button><button id="zout" class="sm">&minus;</button><button id="zfit" class="sm">fit</button><button id="zin" class="sm">+</button></span></div>
   <div id="stripbox"></div>
   <div id="nocrop" class="mut" hidden>No strip is attached to this line. Use "Wrong strip?" to choose one, or look at the page on the right.</div>
   <div id="croppanel" hidden>
    <div class="row" style="margin-bottom:5px"><b>Which printed strip is this line?</b><span class="mut">Click the right one.</span>
     <span style="margin-left:auto" class="row"><button class="sm" id="cpick">More strips&hellip;</button><button class="sm" id="cnone">This line has no strip</button><button class="sm" id="cclose">Close</button></span></div>
    <div class="tiles" id="tiles"></div>
    <div id="cask" hidden style="margin-top:8px"></div>
   </div>
  </div>
  <div class="card" id="editcard">
   <div id="brkbox"></div>
   <div id="editor" class="ed fa" dir="rtl" lang="fa"></div>
   <div id="tb" hidden>
    <div id="tbhead" class="row" style="margin-bottom:5px"></div>
    <textarea id="tbin" class="fa" dir="rtl" lang="fa" rows="1" spellcheck="false" autocomplete="off"></textarea>
    <div class="row" style="margin-top:5px"><button class="sm" id="tbzw">Insert ZWNJ (Alt+Z)</button><button class="sm" id="tbunr">Insert &#10214;&#1567;&#10215; (Alt+U)</button>
     <button class="sm" id="tbconv" title="Latin or Arabic-Indic digits to Persian digits; Arabic ي and ك to Persian ی and ک. Only when you press it.">Convert digits, ي, ك to Persian</button>
     <span id="tbalts" class="row"></span></div>
    <div class="row" style="margin-top:6px"><button id="tbok" class="pri">Use this (Enter)</button><button id="tbcancel">Cancel (Esc)</button><span id="tbextra" class="row"></span></div>
   </div>
   <div id="nav">
    <div class="row"><button id="bprev">&#9664; Previous (K)</button><button id="bnext">Next (J) &#9654;</button><button id="bunres">Next unresolved</button>
     <button id="baccept" class="pri" style="margin-left:auto">Accept line (Enter)</button><button id="breopen">Re-open line</button></div>
    <div class="row"><button class="sm" id="bedit">Edit whole line (E)</button><button class="sm" id="bjoin">Join with next line</button>
     <button class="sm" id="binsa">Insert empty line above</button><button class="sm" id="binsb">Insert empty line below</button><button class="sm" id="bdel">Delete this line</button></div>
   </div>
  </div>
 </section>
 <section id="pagecol" class="col"><div class="card col" style="flex:1;min-height:0">
  <div class="row"><b>Page</b><span class="mut" style="font-size:12px">the current line is outlined</span>
  <span style="margin-left:auto" class="row"><button class="sm" id="pzo">&minus;</button><button class="sm" id="pzi">+</button></span></div>
  <div id="pgwrap"><div id="pgbox"><img id="pgimg" alt="the page"><svg id="pgsvg" preserveAspectRatio="none"></svg></div></div></div></section>
 <section id="listcol" class="col"><div class="card col" style="flex:1;min-height:0">
  <div class="row"><b>All lines</b><span class="mut" id="listinfo" style="font-size:12px"></span><button class="sm" id="showign" style="margin-left:auto" hidden></button></div>
  <div id="list"></div></div></section>
</div>
<div id="toasts"></div><div id="down"></div>
<div class="modal" id="help"><div class="box">
 <div class="row"><b style="font-size:18px">How this works</b><button id="helpx" style="margin-left:auto">Close (Esc)</button></div>
 <p>Each line of the page is checked once. Read the line against the straightened strip, resolve the orange spots, then press <kbd>Enter</kbd> to accept it. Every action is saved at once to the files in the truth folder; you never edit a text file.</p>
 <p><b>Symbols.</b> An orange group is a spot the readers disagreed on: each button is one reading (all may be wrong). <i class="sp">&#9251;</i> is a space, <i class="zw">ZWNJ</i> is a half-space (zero-width non-joiner). Two readings with the same letters but different spaces are a word-join question: look at how it was typed. "Nothing here" removes the word and its space. "Could not be read" / <b>Unreadable</b> keeps <b>&#10214;&#1567;&#10215;</b> in the final text, for a word you cannot read either.</p>
 <table class="k">
  <tr><td><kbd>1</kbd>&ndash;<kbd>9</kbd></td><td>choose that reading for the highlighted spot (for a disputed line break: 1 = one line, 2 = two lines)</td></tr>
  <tr><td><kbd>T</kbd></td><td>type something else; if a reading is already chosen for the spot, the box starts from it so you only edit it (<kbd>Alt+Z</kbd> inserts ZWNJ, <kbd>Alt+U</kbd> inserts &#10214;&#1567;&#10215;, <kbd>Enter</kbd> uses it, <kbd>Esc</kbd> cancels)</td></tr>
  <tr><td><kbd>U</kbd> / <kbd>N</kbd></td><td>unreadable / nothing here</td></tr>
  <tr><td><kbd>Tab</kbd> <kbd>Shift+Tab</kbd></td><td>next / previous spot on the line</td></tr>
  <tr><td><kbd>Enter</kbd></td><td>accept the line (when no spot is left) and go to the next line</td></tr>
  <tr><td><kbd>J</kbd> <kbd>K</kbd></td><td>next / previous line</td></tr>
  <tr><td><kbd>E</kbd></td><td>edit the whole line as text (spots still unresolved stay as &#10214;&hellip;&#10215;)</td></tr>
  <tr><td><kbd>Z</kbd> or <kbd>Ctrl/Cmd+Z</kbd></td><td>undo the last action, again and again</td></tr>
 </table>
 <p>Click any word to correct it, or to start a new line at it. Unmarked words can be wrong too: read them.</p>
 <p><b>Writing rules.</b> Write what is printed, as typed: keep word joins and splits; add no ZWNJ of your own; leave out stretched letters (kashida); Persian digits, ی and ک; keep the page-number line; do not write stamps, signatures or handwriting. One printed line per line.</p>
 <p><b>One printed line at a time.</b> Lines follow the printed strips found on the page, not the readers' own line breaks: where readers broke a line differently, their words were put back on the strip they fit. If a word still sits on the wrong line, click it and use "Start a new line at this word", or "Join with next line".</p>
 <p><b>Strips.</b> The strip above each line was matched automatically. If it is the wrong one, press "Wrong strip?" and click the right one; you are then asked whether it is for this line only or for this line and all lines after it. A grey dashed row in the list marks a printed strip that no line uses: add a line for it (for example the page number) or mark it "not text".</p>
 <p><b>Done.</b> A line is done only after you press accept; any later edit re-opens it. A page is audited when every line is done. Until then the exported <code>000N.lines.txt</code> still contains the unresolved markers and the page shows NOT FINISHED.</p>
</div></div>
<div class="modal" id="picker"><div class="box">
 <div class="row"><b style="font-size:18px">Pick the strip for this line</b><button id="pickx" style="margin-left:auto">Close (Esc)</button></div>
 <p class="mut">Strips in page order, top to bottom.</p><div class="pick" id="pickgrid"></div>
</div></div>
</div></div>
<script>
"use strict";
var PORT="__PORT__", CMD="__CMD__";
var S={page:null,doc:null,undo:0,cur:{line:0,mark:null},tb:null,zmul:1,pgz:1,down:false,pages:[],showIgn:false,focusCrop:null,sel:null};
function $(i){return document.getElementById(i)}
function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}
function vis(s){var o="";for(var c of s){if(c===" ")o+='<i class="sp" title="space">␣</i><wbr>';
 else if(c==="‌")o+='<i class="zw" title="ZWNJ (zero-width non-joiner)">ZWNJ</i>';
 else if(c===" ")o+='<i class="zw">NBSP</i>';else if(c==="‏")o+='<i class="zw">RLM</i>';else if(c==="‎")o+='<i class="zw">LRM</i>';
 else o+=esc(c)}return o}
/* ---------- layout (real window, or a ?w=&h= scaled preview) ---------- */
var Q=new URLSearchParams(location.search),OW=+Q.get("w")||0,OH=+Q.get("h")||0,EW=0;
function layout(){var app=$("app"),rw=innerWidth,rh=innerHeight,W=OW||rw,H,k=1;
 if(OW){k=rw/OW;if(OH)k=Math.min(k,rh/OH);H=OH||rh/k;document.documentElement.classList.add("preview");
  app.style.width=W+"px";app.style.transform="scale("+k+")"}
 else{H=rh;app.style.width="100%"}
 EW=W;app.style.height=H+"px";document.documentElement.style.setProperty("--vh",H+"px");
 app.className=W>=1300?"L-wide":(W>=900?"L-mid":"L-narrow");
 if(S.doc)renderStrip()}
/* ---------- toasts ---------- */
var tmr=null;
function toast(msg,kind,btn){var t=$("toasts");if(!msg){t.innerHTML="";return}
 t.innerHTML="";var d=document.createElement("div");d.className="toast "+(kind||"info");d.appendChild(document.createTextNode(msg));
 if(btn){var b=document.createElement("button");b.textContent=btn.label;b.className="sm";b.onclick=function(){t.innerHTML="";btn.fn()};d.appendChild(b)}
 var x=document.createElement("button");x.textContent="×";x.className="sm";x.onclick=function(){t.innerHTML=""};d.appendChild(x);t.appendChild(d);
 clearTimeout(tmr);if(kind==="info")tmr=setTimeout(function(){t.innerHTML=""},4000)}
function notice(msg,kind,btn){toast(msg,kind,btn)}
/* ---------- network ---------- */
var chain=Promise.resolve();
function serial(fn){var p=chain.then(fn);chain=p.catch(function(){});return p}
function setDown(d){if(d===S.down)return;S.down=d;var b=$("down");
 if(d){b.style.display="block";b.innerHTML="The tool's server is not running (this browser could not reach http://127.0.0.1:"+PORT+"). Nothing is lost: every earlier action was already saved to disk. To restart it, open Terminal and run:<code>"+esc(CMD)+"</code>Then reload this page (your place is remembered)."}
 else{b.style.display="none"}}
async function call(method,path,body){var r;
 try{r=await fetch(path,{method:method,headers:{"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined,cache:"no-store"})}
 catch(e){setDown(true);throw {down:true}}
 setDown(false);var d=null;try{d=await r.json()}catch(e){}
 if(!d)throw {msg:"The server sent an unreadable reply (HTTP "+r.status+")."};
 d.http=r.status;return d}
function accept(d){
 if(d.http===409){S.doc=d.doc;toast("This page was changed in another tab, so it was reloaded. Your last action was NOT applied; repeat it if it is still needed.","warn");S.tb=null;render();return null}
 if(!d.ok)return d;
 S.doc=d.doc;if(d.undo_depth!==undefined)S.undo=d.undo_depth;return d}
function doOp(op,args,opts){return serial(async function(){var d;
 try{d=await call("POST","/api/op",{page:S.page,op:op,args:args,rev:S.doc.revision})}catch(e){if(!e.down)toast(e.msg||String(e),"err");return null}
 var r=accept(d);if(r===null)return null;if(!r.ok){if(opts&&opts.errfn&&opts.errfn(r.error))return null;toast(r.error,"err");return null}
 return r})}
function doUndo(){return serial(async function(){var d;
 try{d=await call("POST","/api/undo",{page:S.page,rev:S.doc.revision})}catch(e){if(!e.down)toast(e.msg||String(e),"err");return null}
 var r=accept(d);if(r===null)return null;if(!r.ok){toast(r.error,"err");return null}
 S.tb=null;S.cur.line=Math.min(S.doc.cursor.line,S.doc.lines.length-1);S.cur.mark=firstUnres(curLine());
 closeCropPanel();toast("Undid: "+r.undone,"info");render();return r})}
function setCursor(){call("POST","/api/op",{page:S.page,op:"cursor",args:{line:S.cur.line,mark:S.cur.mark},rev:null}).catch(function(){})}
async function loadPages(){try{var d=await call("GET","/api/pages");S.pages=d.pages||[];renderTabs()}catch(e){}}
async function loadPage(p){try{var d=await call("GET","/api/page?page="+p);
 if(!d.ok){toast(d.error,"err");return false}
 S.page=p;S.doc=d.doc;S.undo=d.undo_depth;S.tb=null;S.focusCrop=null;store("ta_page",String(p));
 S.cur={line:Math.min(d.doc.cursor.line||0,Math.max(0,d.doc.lines.length-1)),mark:null};S.cur.mark=firstUnres(curLine());
 closeCropPanel();render();loadPages();
 var mg=d.doc.migrated;if(mg&&store("ta_mig_"+p)!==mg.at){store("ta_mig_"+p,mg.at);
  toast("This page was rebuilt so that every printed strip is one line and choices are word by word. Kept: "+mg.lines_kept_accepted+" accepted line(s), "+mg.choices_carried+" choice(s). To redo: "+(mg.accepted_lines_to_redo.length)+" accepted line(s), "+(mg.choices_not_carried.length)+" choice(s). The earlier state is saved beside it.","warn")}
 return true}catch(e){if(!e.down)toast(e.msg||String(e),"err");return false}}
/* ---------- helpers ---------- */
function curLine(){return S.doc&&S.doc.lines[S.cur.line]||null}
function spots(L){return L?L.segs.filter(function(s){return s.t==="mark"||s.t==="brk"}):[]}
function isUnres(s){return s.t==="brk"||(s.t==="mark"&&s.choice===null)}
function firstUnres(L){var a=spots(L).filter(isUnres);return a.length?a[0].id:null}
function nextSpot(L,from,dir){var a=spots(L);if(!a.length)return null;var i=a.findIndex(function(s){return s.id===from});
 var un=a.filter(isUnres);if(!un.length)return null;
 for(var k=1;k<=a.length;k++){var j=((i<0?(dir>0?-1:0):i)+dir*k+a.length*2)%a.length;if(isUnres(a[j]))return a[j].id}return un[0].id}
function markText(s){var c=s.choice;if(!c)return null;
 if(c.k==="alt"){var a=s.alts[c.i];return a==="∅"||a===""?"":(a==="…"?"⟦؟⟧":a)}
 if(c.k==="custom")return c.v;if(c.k==="unreadable")return "⟦؟⟧";return ""}
function yAt(c,x){var xs=c.xs,ys=c.ys,n=xs.length;if(n===1)return ys[0];
 if(x<=xs[0])return ys[0]+(ys[1]-ys[0])/(xs[1]-xs[0]||1)*(x-xs[0]);
 if(x>=xs[n-1])return ys[n-1]+(ys[n-1]-ys[n-2])/(xs[n-1]-xs[n-2]||1)*(x-xs[n-1]);
 for(var i=1;i<n;i++){if(x<=xs[i]){var t=(x-xs[i-1])/(xs[i]-xs[i-1]||1);return ys[i-1]+t*(ys[i]-ys[i-1])}}return ys[n-1]}
function cropById(id){return id?S.doc.crops.find(function(c){return c.id===id}):null}
function cropOf(L){return L?cropById(L.crop):null}
function usedCrops(){var u={};S.doc.lines.forEach(function(l,i){if(l.crop)u[l.crop]=i+1});return u}
function thumbSrc(c){return c.file?"/crop?page="+S.page+"&file="+encodeURIComponent(c.file):"/strip?page="+S.page+"&crop="+c.id}
/* slim rows for strips no line uses, in page order */
function unmatchedBefore(){ // list of {crop, at} where at = index of the line the row sits before
 var d=S.doc,used=usedCrops(),ign=d.ignored||[],out=[],emitted={};
 var un=d.crops.filter(function(c){return !used[c.id]&&(S.showIgn||ign.indexOf(c.id)<0)});
 var lastOrder=-1,pending=un.slice();
 d.lines.forEach(function(l,i){var c=cropById(l.crop);if(!c)return;
  pending=pending.filter(function(u){if(u.order<c.order&&u.order>lastOrder){out.push({crop:u,at:i});return false}return true});lastOrder=Math.max(lastOrder,c.order)});
 pending.forEach(function(u){out.push({crop:u,at:d.lines.length})});return out}
/* ---------- render ---------- */
function render(){renderTabs();renderStatus();renderStrip();renderEditor();renderList();renderPage();
 $("undo").disabled=!S.undo;var L=curLine();var acc=L&&L.state==="accepted";
 $("breopen").style.display=acc?"":"none";$("baccept").style.display=acc?"none":""}
function renderTabs(){var h="";S.pages.forEach(function(p){
 h+='<button class="tab '+(p.page===S.page?"on":"")+'" data-p="'+p.page+'">'+esc(p.label)+(p.error?'<small>error</small>':'<small>'+p.markers_left+' spots &middot; '+p.lines_left+' lines'+(p.status==="audited"?" ✓":"")+'</small>')+'</button>'});
 $("tabs").innerHTML=h}
function renderStatus(){var d=S.doc;if(!d)return;var s=d.summary;
 $("status").innerHTML='spots left <b>'+s.markers_left+'</b> &middot; lines left <b>'+s.lines_left+'</b>/'+s.lines+' &middot; '+
 (s.audited?'<span class="badge fin">AUDITED: export is final</span>':'<span class="badge nf">NOT FINISHED: export still has draft markers</span>')+
 ' <span class="mut">'+esc(d.export.file)+'</span>'}
function stripGeom(c,cw){var g=c.strip,ov=30,Z;
 var fit1=cw/g.w;
 var base=Math.min(1.15,Math.max(fit1,cw*2/(g.w+2*ov)));
 Z=Math.max(0.3,Math.min(3,base*S.zmul));
 var R=Math.max(1,Math.ceil((g.w-ov)/(cw/Z-ov)));if(R>6){R=6;Z=cw/((g.w-ov)/R+ov)}
 return {Z:Z,R:R,ov:ov,src:cw/Z}}
function renderStrip(){var L=curLine(),c=cropOf(L),d=S.doc,box=$("stripbox");
 $("lineinfo").textContent=L?("Line "+(S.cur.line+1)+" of "+d.lines.length):"No lines";
 $("lineflags").innerHTML=L?(L.state==="accepted"?'<span class="badge fin">accepted</span>':(L.unres?'<span class="badge nf">'+L.unres+' spot'+(L.unres>1?'s':'')+' left</span>':'<span class="badge nf">read it, then accept</span>')):"";
 var has=!!c;box.style.display=has?"":"none";$("nocrop").hidden=has;
 var order=has?(d.crops.indexOf(c)+1):0;
 $("cropinfo").textContent=c?("strip "+order+"/"+d.crops.length+(L.crop_src==="owner"?" (set by you)":"")):(L&&L.crop_src==="owner"?"no strip (set by you)":"no strip matched");
 if(!has){box.innerHTML="";return}
 var cw=box.clientWidth||600,g=c.strip,G=stripGeom(c,cw),url="/strip?page="+S.page+"&crop="+c.id,h="";
 var act=L.pos[S.cur.mark];
 var bandH=g.band_bottom-g.band_top,vy0=Math.max(0,g.band_top-0.25*bandH),vy1=Math.min(g.h,g.band_bottom+0.25*bandH);
 var rowH=Math.round((vy1-vy0)*G.Z);
 for(var i=0;i<G.R;i++){var xr=g.w-i*(G.src-G.ov),xl=xr-G.src;
  h+='<div class="srow" style="height:'+rowH+'px"><img src="'+url+'" style="left:'+(-xl*G.Z)+'px;top:'+(-vy0*G.Z)+'px;width:'+(g.w*G.Z)+'px;height:'+(g.h*G.Z)+'px">'+
   '<div class="dim" style="top:0;height:'+((g.band_top-vy0)*G.Z)+'px"></div><div class="dim" style="top:'+((g.band_bottom-vy0)*G.Z)+'px;height:'+((vy1-g.band_bottom)*G.Z)+'px"></div>'+
   (G.R>1?'<span class="tag">'+(i+1)+'/'+G.R+'</span>':'');
  spots(L).forEach(function(s,k){var f=L.pos[s.id];if(f===undefined||s.t==="brk")return;
   var u=g.margin+(c.x1-c.x0)*(1-f);
   var row=0;for(var r=0;r<G.R;r++){if(u>=g.w-r*(G.src-G.ov)-G.src+12){row=r;break}row=r}
   if(row!==i)return;
   h+='<div class="ptr'+(s.id===S.cur.mark?" act":"")+'" style="left:'+((u-xl)*G.Z)+'px" title="approximate position of spot '+(k+1)+' (proportional, not measured)"></div>'});
  h+='</div>'}
 box.innerHTML=h}
function renderEditor(){var L=curLine(),ed=$("editor");if(!L){ed.innerHTML="";$("brkbox").innerHTML="";return}
 var bb="",d=S.doc,pr=d.pairs[String(S.cur.line)];
 var hasb=L.segs.some(function(s){return s.t==="brk"});
 if(hasb&&pr){var other=pr[1]===null?null:(pr[0]===S.cur.line?pr[1]:pr[0]);
  bb='<div class="brk"><b>Disputed line break.</b> The readers disagreed whether this printed line ends here. '+
  (other===null?'<button class="sm" data-act="brk" data-w="two">Remove the mark</button>':
  'Lines '+(pr[0]+1)+' and '+(pr[1]+1)+' are shown as two lines. Is the printed text ...'+
  '<div class="other fa">'+esc(d.lines[pr[0]].out)+'<br>'+esc(d.lines[pr[1]].out)+'</div>'+
  '<div class="row"><button class="alt" data-act="brk" data-w="one"><b class="k" style="font:700 12px system-ui;background:var(--ac);color:#fff;border-radius:999px;padding:0 6px;margin-right:6px">1</b>One printed line (join)</button> <button class="alt" data-act="brk" data-w="two"><b class="k" style="font:700 12px system-ui;background:var(--ac);color:#fff;border-radius:999px;padding:0 6px;margin-right:6px">2</b>Two printed lines (keep apart)</button></div>')+'</div>'}
 $("brkbox").innerHTML=bb;
 var h="";
 L.segs.forEach(function(s,si){
  if(s.t==="text"){var re=/( +)|([^ ]+)/g,m;while((m=re.exec(s.v))!==null){
    if(m[2])h+='<span class="w" data-seg="'+si+'" data-s="'+m.index+'" data-e="'+(m.index+m[0].length)+'">'+vis(m[2]).replace(/<wbr>/g,"")+'</span>';else h+=esc(m[1])}}
  else if(s.t==="mark"){h+=markHtml(s,si,L)}});
 if(!L.segs.length)h='<span class="mut" style="font-size:15px;direction:ltr;display:inline-block">(empty line: press E to type its text, or delete the line)</span>';
 ed.innerHTML=h;
 var act=ed.querySelector(".mk.act");if(act&&act.scrollIntoView)act.scrollIntoView({block:"nearest",inline:"nearest"})}
function markHtml(s,si,L){var un=s.choice===null,act=s.id===S.cur.mark;
 if(!un&&!act){var t=markText(s);return '<span class="mk done" data-mid="'+s.id+'"><button class="chosen fa" data-act="focus" title="click to change">'+(t===""?'<span style="font:600 14px system-ui;color:var(--mut)">(nothing)</span>':vis(t))+'</button></span>'}
 var h='<span class="mk '+(un?"":"done ")+(act?"act":"")+'" data-mid="'+s.id+'"><span class="alts">';
 if(!s.alts.length)h+='<span style="font:600 14px system-ui;direction:ltr;padding:4px 8px">No reader could read this word</span>';
 s.alts.forEach(function(a,i){var on=s.choice&&s.choice.k==="alt"&&s.choice.i===i;var lab;var cls="alt"+(on?" on":"");
  if(a==="∅"||a===""){lab='<span class="plain">nothing here (remove)</span>';cls+=" gap"}
  else if(a==="…"){lab='<span class="plain">could not be read (keep ⟦؟⟧)</span>';cls+=" gap"}
  else lab=vis(a);
  h+='<button class="'+cls+'" data-act="alt" data-i="'+i+'">'+(i<9?'<b class="k">'+(i+1)+'</b>':'')+lab+'</button>'});
 h+='</span>';
 if(s.choice&&s.choice.k==="custom")h+='<div class="mknote">Typed by you: <span class="fa" style="font-size:19px;direction:rtl;unicode-bidi:isolate">'+vis(s.choice.v)+'</span></div>';
 if(s.note)h+='<div class="mknote">'+esc(s.note)+'</div>';
 if(act||!s.alts.length){h+='<div class="mkact"><button class="sm" data-act="type">Type it (T)</button><button class="sm" data-act="unreadable">Unreadable (U)</button><button class="sm" data-act="nothing">Nothing here (N)</button><button class="sm" data-act="newline" title="start a new line at this spot">New line from here</button></div>'}
 return h+'</span>'}
function renderList(){var h="",d=S.doc,um=unmatchedBefore(),k=0,ign=d.ignored||[];
 function unmRow(u){var c=u.crop,ig=ign.indexOf(c.id)>=0;
  return '<div class="unm'+(S.focusCrop===c.id?" f":"")+'" data-uc="'+c.id+'"><img src="'+thumbSrc(c)+'" data-uf="'+c.id+'" title="show on the page"><span class="m">'+(ig?'ignored: not text':'Printed strip with no text line here')+'</span>'+
   (ig?'<button class="sm" data-ur="'+c.id+'">Restore</button>':'<button class="sm" data-ua="'+c.id+'" data-at="'+u.at+'">Add a line for it</button><button class="sm" data-ui="'+c.id+'">Not text (ignore)</button>')+'</div>'}
 d.lines.forEach(function(L,i){
  while(k<um.length&&um[k].at===i){h+=unmRow(um[k]);k++}
  var st=L.state==="accepted"?'<span class="st ok">✓ done</span>':(L.unres?'<span class="st wn">● '+L.unres+' left</span>':'<span class="st wn">○ to accept</span>');
  var tx=esc(L.out).replace(/(⟦[^⟧]*⟧)/g,"<mark>$1</mark>");
  h+='<div class="li fa'+(i===S.cur.line?" cur":"")+'" data-l="'+i+'"><span class="n">'+(i+1)+'</span>'+st+'<span class="tx">'+tx+'</span></div>'});
 while(k<um.length){h+=unmRow(um[k]);k++}
 var lst=$("list"),keep=lst.scrollTop;lst.innerHTML=h;lst.scrollTop=keep;
 $("listinfo").textContent=d.lines.length+" lines";
 var nign=ign.length;$("showign").hidden=!nign;$("showign").textContent=(S.showIgn?"Hide":"Show")+" "+nign+" ignored";
 var c=lst.querySelector(".li.cur");if(c){var t=c.offsetTop-lst.offsetTop;if(t<lst.scrollTop+20||t>lst.scrollTop+lst.clientHeight-40)lst.scrollTop=Math.max(0,t-lst.clientHeight/3)}}
function renderPage(){var d=S.doc,img=$("pgimg"),url="/img?page="+S.page;
 if(img.getAttribute("data-u")!==url){img.setAttribute("data-u",url);img.src=url}
 $("pgbox").style.width=(S.pgz*100)+"%";
 var W=d.image.width,H=d.image.height,svg=$("pgsvg");svg.setAttribute("viewBox","0 0 "+W+" "+H);
 var L=curLine(),cc=cropById(S.focusCrop)||cropOf(L),h="";
 d.crops.forEach(function(c){if(c===cc)return;h+='<polygon points="'+poly(c)+'" fill="none" stroke="rgba(31,95,191,.35)" stroke-width="'+(W/700)+'"/>'});
 if(cc)h+='<polygon points="'+poly(cc)+'" fill="rgba(255,200,0,.25)" stroke="#d9480f" stroke-width="'+(W/280)+'"/>';
 svg.innerHTML=h;
 if(cc){var yc=yAt(cc,(cc.x0+cc.x1)/2),wrap=$("pgwrap"),sc=img.clientWidth/W;
  if(sc>0)wrap.scrollTop=yc*sc-wrap.clientHeight*0.4;else img.onload=function(){renderPage()}}}
function poly(c){var n=24,t=[],b=[];for(var i=0;i<=n;i++){var x=c.x0+(c.x1-c.x0)*i/n,y=yAt(c,x);t.push(x.toFixed(1)+","+(y-c.strip.up_p).toFixed(1));b.push(x.toFixed(1)+","+(y+c.strip.dn_p).toFixed(1))}
 return t.concat(b.reverse()).join(" ")}
/* ---------- strip correction (D) ---------- */
function openCropPanel(){var L=curLine();if(!L)return;S.sel=null;$("croppanel").hidden=false;$("cask").hidden=true;renderTiles()}
function closeCropPanel(){S.sel=null;$("croppanel").hidden=true}
function renderTiles(){var d=S.doc,L=curLine(),cur=cropOf(L),base;
 if(cur)base=cur.order;else{base=-1;for(var i=S.cur.line-1;i>=0;i--){var p=cropOf(d.lines[i]);if(p){base=p.order;break}}base+=1}
 var h="",names={"-1":"previous strip (up)","0":cur?"this line's strip now":"next strip","1":"next strip (down)"};
 [-1,0,1].forEach(function(o){var c=d.crops[base+o];if(!c){h+='<div></div>';return}
  var isCur=cur&&c.id===cur.id;h+='<div class="t'+(isCur?" cur":"")+(S.sel===c.id?" sel":"")+'" data-t="'+c.id+'"><div>'+(isCur?"This line's strip now":(o<0?"Previous strip (up)":(o>0?"Next strip (down)":"Strip")))+' &middot; #'+(c.order+1)+'</div><img src="'+thumbSrc(c)+'"></div>'});
 $("tiles").innerHTML=h}
function askScope(cid){var d=S.doc,L=curLine(),cur=cropOf(L),c=cropById(cid);S.sel=cid;renderTiles();
 var h='<b>Use strip #'+(c.order+1)+' for ...</b> <span class="row" style="display:inline-flex"><button class="pri" data-sc="one">this line only</button>';
 if(cur){var dd=c.order-cur.order,after=d.lines.length-S.cur.line-1;
  h+='<button class="pri" data-sc="after">this line and all '+after+' lines after it (each moves '+(dd>0?"down":"up")+' by '+Math.abs(dd)+')</button>'}
 h+='<button data-sc="cancel">Cancel</button></span>';
 $("cask").innerHTML=h;$("cask").hidden=false}
/* ---------- type box ---------- */
function openTb(mode,o){var L=curLine();S.tb=Object.assign({mode:mode},o||{});var tb=$("tb");tb.hidden=false;
 var head="",val="",extra="",alts="";
 if(mode==="choose"){head="Type what is printed for this spot";var s=spots(L).find(function(x){return x.id===o.mark});
  /* start from what is already chosen for this spot (typed text, or the selected reading), so it is edited, not retyped */
  val="";if(s&&s.choice){if(s.choice.k==="custom")val=s.choice.v;else if(s.choice.k==="alt"){var ca=s.alts[s.choice.i];if(ca!=="∅"&&ca!=="…"&&ca!=="")val=ca}}
  if(val)head="Edit the reading you chose for this spot";
  if(s)s.alts.forEach(function(a,i){if(a!=="∅"&&a!=="…"&&a!=="")alts+='<button class="sm" data-tbalt="'+i+'" title="copy this reading into the box">start from '+(i+1)+'</button>'})}
 else if(mode==="word"){head="Correct this word";val=o.text;extra='<button class="sm" data-tbx="split">Start a new line at this word</button><button class="sm" data-tbx="delword">Delete this word</button>'}
 else{head="Whole line as text (unresolved spots stay as ⟦...⟧; resolved ones become plain text)";val=L.out}
 $("tbhead").innerHTML="<b>"+esc(head)+"</b>";$("tbalts").innerHTML=alts;$("tbextra").innerHTML=extra;
 var ta=$("tbin");ta.rows=mode==="line"?3:1;ta.value=val;ta.focus();ta.setSelectionRange(val.length,val.length)}
function closeTb(){S.tb=null;$("tb").hidden=true;focusEditor()}
function focusEditor(){if(document.activeElement&&document.activeElement.blur&&/TEXTAREA|INPUT/.test(document.activeElement.tagName))document.activeElement.blur()}
function insertAtCursor(txt){var ta=$("tbin");ta.setRangeText(txt,ta.selectionStart,ta.selectionEnd,"end");ta.focus()}
function toPersian(s){var o="";for(var c of s){var k=c.charCodeAt(0);
 if(k>=48&&k<=57)o+=String.fromCharCode(0x06F0+k-48);else if(k>=0x660&&k<=0x669)o+=String.fromCharCode(0x06F0+k-0x660);
 else if(c==="ي")o+="ی";else if(c==="ك")o+="ک";else o+=c}return o}
async function commitTb(){var t=S.tb,L=curLine();if(!t||!L)return;var v=$("tbin").value.replace(/[\r\n]+/g," ");
 var r=null;
 if(t.mode==="choose"){r=await chooseSpot(t.mark,{k:"custom",v:v})}
 else if(t.mode==="word"){r=await doOp("edit",{line:S.cur.line,lid:L.id,seg:t.seg,start:t.start,end:t.end,text:v});if(r)S.cur.mark=firstUnres(curLine())}
 else{r=await doOp("set_text",{line:S.cur.line,lid:L.id,text:v});if(r)S.cur.mark=firstUnres(curLine())}
 if(r){S.tb=null;$("tb").hidden=true;render();focusEditor()}}
/* ---------- actions ---------- */
async function chooseSpot(mid,choice){var L=curLine();var r=await doOp("choose",{line:S.cur.line,lid:L.id,mark:mid,choice:choice});
 if(r){var L2=curLine();S.cur.mark=nextSpot(L2,mid,1);render();setCursor()}return r}
async function selectLine(i){if(!S.doc||i<0||i>=S.doc.lines.length)return;S.tb=null;$("tb").hidden=true;closeCropPanel();S.focusCrop=null;S.cur.line=i;S.cur.mark=firstUnres(curLine());render();setCursor();
 if(EW<900){var sh=$("cropcard");if(sh&&sh.scrollIntoView)sh.scrollIntoView({block:"start"})}}
async function acceptLine(force){var L=curLine();if(!L)return;
 if(L.unres){S.cur.mark=firstUnres(L);render();toast("Resolve the spots on this line first.","warn");return}
 var r=await doOp("accept",{line:S.cur.line,lid:L.id,allow_gap_marks:!!force},{errfn:function(e){
   if(e.indexOf("NEEDS_CONFIRM")===0){toast(e.replace("NEEDS_CONFIRM: ",""),"warn",{label:"Accept anyway",fn:function(){acceptLine(true)}});return true}return false}});
 if(!r)return;
 var n=S.doc.lines.length,j=-1;for(var k=1;k<=n;k++){var x=(S.cur.line+k)%n;if(S.doc.lines[x].state!=="accepted"){j=x;break}}
 if(j<0){render();toast(S.doc.summary.audited?"Every line is accepted: this page is audited and its export is final.":"All lines accepted.","info");return}
 await selectLine(j)}
function actOnMark(act,mid,el){var L=curLine();
 if(act==="focus"){S.cur.mark=mid;render();setCursor();return}
 if(act==="alt"){chooseSpot(mid,{k:"alt",i:+el.getAttribute("data-i")});return}
 if(act==="type"){S.cur.mark=mid;render();openTb("choose",{mark:mid});return}
 if(act==="unreadable"){chooseSpot(mid,{k:"unreadable"});return}
 if(act==="nothing"){chooseSpot(mid,{k:"empty"});return}
 if(act==="newline"){var si=L.segs.findIndex(function(s){return s.id===mid});doOp("split",{line:S.cur.line,lid:L.id,seg:si,offset:0}).then(function(r){if(r){S.cur.mark=firstUnres(curLine());render()}});return}}
/* ---------- events ---------- */
$("tabs").addEventListener("click",function(e){var b=e.target.closest("[data-p]");if(b)loadPage(+b.getAttribute("data-p"))});
$("editor").addEventListener("click",function(e){var L=curLine();if(!L)return;
 var w=e.target.closest(".w");if(w){openTb("word",{seg:+w.getAttribute("data-seg"),start:+w.getAttribute("data-s"),end:+w.getAttribute("data-e"),text:L.segs[+w.getAttribute("data-seg")].v.slice(+w.getAttribute("data-s"),+w.getAttribute("data-e"))});return}
 var b=e.target.closest("[data-act]");if(!b)return;var m=b.closest(".mk");actOnMark(b.getAttribute("data-act"),m?+m.getAttribute("data-mid"):null,b)});
$("brkbox").addEventListener("click",function(e){var b=e.target.closest("[data-act=brk]");if(!b)return;var L=curLine();
 doOp("brk",{line:S.cur.line,lid:L.id,which:b.getAttribute("data-w")}).then(function(r){if(r){S.cur.line=Math.min(S.cur.line,S.doc.lines.length-1);S.cur.mark=firstUnres(curLine());render();setCursor()}})});
$("list").addEventListener("click",function(e){
 var ua=e.target.closest("[data-ua]");if(ua){var at=+ua.getAttribute("data-at");doOp("add_line_for_crop",{at:at,crop:ua.getAttribute("data-ua")}).then(function(r){if(r){S.cur.line=at;S.cur.mark=null;S.focusCrop=null;render();setCursor();openTb("line",{})}});return}
 var ui=e.target.closest("[data-ui]");if(ui){doOp("ignore_crop",{crop:ui.getAttribute("data-ui"),ignore:true}).then(function(r){if(r){toast("Strip marked as not text (Undo brings it back).","info");render()}});return}
 var ur=e.target.closest("[data-ur]");if(ur){doOp("ignore_crop",{crop:ur.getAttribute("data-ur"),ignore:false}).then(function(r){if(r)render()});return}
 var uf=e.target.closest("[data-uf]");if(uf){S.focusCrop=uf.getAttribute("data-uf");renderList();renderPage();return}
 var r=e.target.closest("[data-l]");if(r)selectLine(+r.getAttribute("data-l"))});
$("showign").onclick=function(){S.showIgn=!S.showIgn;renderList()};
$("tb").addEventListener("click",function(e){var b=e.target.closest("[data-tbalt]");if(b){var s=spots(curLine()).find(function(x){return x.id===S.tb.mark});$("tbin").value=s.alts[+b.getAttribute("data-tbalt")];$("tbin").focus();return}
 var x=e.target.closest("[data-tbx]");if(!x)return;var L=curLine(),t=S.tb;
 if(x.getAttribute("data-tbx")==="split"){doOp("split",{line:S.cur.line,lid:L.id,seg:t.seg,offset:t.start}).then(function(r){if(r){S.tb=null;$("tb").hidden=true;S.cur.mark=firstUnres(curLine());render()}})}
 else{doOp("edit",{line:S.cur.line,lid:L.id,seg:t.seg,start:t.start,end:t.end,text:""}).then(function(r){if(r){S.tb=null;$("tb").hidden=true;S.cur.mark=firstUnres(curLine());render()}})}});
$("tbok").onclick=commitTb;$("tbcancel").onclick=closeTb;
$("tbzw").onclick=function(){insertAtCursor("‌")};$("tbunr").onclick=function(){insertAtCursor("⟦؟⟧")};
$("tbconv").onclick=function(){var ta=$("tbin");ta.value=toPersian(ta.value);ta.focus()};
$("tbin").addEventListener("keydown",function(e){
 if(e.isComposing||e.keyCode===229)return;
 if(e.key==="Enter"&&!(S.tb&&S.tb.mode==="line"&&e.shiftKey)){e.preventDefault();commitTb()}
 else if(e.key==="Escape"){e.preventDefault();closeTb()}
 else if(e.altKey&&e.code==="KeyZ"){e.preventDefault();insertAtCursor("‌")}
 else if(e.altKey&&e.code==="KeyU"){e.preventDefault();insertAtCursor("⟦؟⟧")}
 e.stopPropagation()});
$("undo").onclick=doUndo;$("helpb").onclick=function(){$("help").classList.add("on")};$("helpx").onclick=function(){$("help").classList.remove("on")};
$("bprev").onclick=function(){selectLine(S.cur.line-1)};$("bnext").onclick=function(){selectLine(S.cur.line+1)};
$("bunres").onclick=function(){var n=S.doc.lines.length;for(var k=1;k<=n;k++){var x=(S.cur.line+k)%n;if(S.doc.lines[x].unres){selectLine(x);return}}toast("No unresolved spot is left on this page.","info")};
$("baccept").onclick=function(){acceptLine(false)};
$("breopen").onclick=function(){var L=curLine();doOp("reopen",{line:S.cur.line,lid:L.id}).then(function(r){if(r)render()})};
$("bedit").onclick=function(){openTb("line",{})};
$("bjoin").onclick=function(){var L=curLine();doOp("join",{line:S.cur.line,lid:L.id}).then(function(r){if(r){S.cur.mark=firstUnres(curLine());render();setCursor()}})};
$("binsa").onclick=function(){doOp("insert",{at:S.cur.line}).then(function(r){if(r){S.cur.mark=null;render();setCursor();openTb("line",{})}})};
$("binsb").onclick=function(){doOp("insert",{at:S.cur.line+1}).then(function(r){if(r){S.cur.line++;S.cur.mark=null;render();setCursor();openTb("line",{})}})};
$("bdel").onclick=function(){var L=curLine();doOp("delete",{line:S.cur.line,lid:L.id}).then(function(r){if(r){S.cur.line=Math.min(S.cur.line,S.doc.lines.length-1);S.cur.mark=firstUnres(curLine());render();setCursor();toast("Line deleted (Undo brings it back).","info")}})};
$("wrongstrip").onclick=function(){if($("croppanel").hidden)openCropPanel();else closeCropPanel()};
$("cclose").onclick=closeCropPanel;
$("tiles").addEventListener("click",function(e){var t=e.target.closest("[data-t]");if(!t)return;var L=curLine(),cur=cropOf(L);
 if(cur&&cur.id===t.getAttribute("data-t"))return;askScope(t.getAttribute("data-t"))});
$("cask").addEventListener("click",function(e){var b=e.target.closest("[data-sc]");if(!b)return;var sc=b.getAttribute("data-sc"),L=curLine();
 if(sc==="cancel"){S.sel=null;$("cask").hidden=true;renderTiles();return}
 doOp("set_crop",{line:S.cur.line,lid:L.id,crop:S.sel,following:sc==="after"}).then(function(r){if(r){closeCropPanel();render();toast(sc==="after"?"Strips moved for this line and the lines after it (Undo restores them).":"Strip changed for this line (Undo restores it).","info")}})});
$("cnone").onclick=function(){var L=curLine();doOp("set_crop",{line:S.cur.line,lid:L.id,crop:null}).then(function(r){if(r){closeCropPanel();render()}})};
$("cpick").onclick=function(){var d=S.doc,used=usedCrops(),L=curLine(),h="";
 d.crops.forEach(function(c,k){h+='<div class="c'+(L.crop===c.id?" on":"")+'" data-c="'+c.id+'"><div>Strip '+(k+1)+' &middot; '+(used[c.id]?"line "+used[c.id]:"<b>no line</b>")+'</div><img loading="lazy" src="'+thumbSrc(c)+'"></div>'});
 $("pickgrid").innerHTML=h;$("picker").classList.add("on")};
$("pickgrid").addEventListener("click",function(e){var c=e.target.closest("[data-c]");if(!c)return;$("picker").classList.remove("on");askScopeAfterPick(c.getAttribute("data-c"))});
function askScopeAfterPick(cid){var L=curLine();if(cropOf(L)&&cropOf(L).id===cid)return;$("croppanel").hidden=false;askScope(cid)}
$("pickx").onclick=function(){$("picker").classList.remove("on")};
$("zin").onclick=function(){S.zmul=Math.min(2.5,S.zmul*1.25);renderStrip()};
$("zout").onclick=function(){S.zmul=Math.max(0.5,S.zmul/1.25);renderStrip()};
$("zfit").onclick=function(){S.zmul=1;renderStrip()};
$("pzi").onclick=function(){S.pgz=Math.min(3,S.pgz*1.25);renderPage()};$("pzo").onclick=function(){S.pgz=Math.max(1,S.pgz/1.25);renderPage()};
window.addEventListener("resize",layout);
document.addEventListener("click",function(e){var b=e.target.closest&&e.target.closest("button");if(b&&b.blur)setTimeout(function(){b.blur()},0)});
var kq=Promise.resolve();
function keyJob(fn){kq=kq.then(fn).catch(function(){})}
document.addEventListener("keydown",function(e){
 var tg=e.target&&e.target.tagName;if(tg==="TEXTAREA"||tg==="INPUT")return;
 if($("help").classList.contains("on")||$("picker").classList.contains("on")){if(e.key==="Escape"){$("help").classList.remove("on");$("picker").classList.remove("on")}return}
 if(!S.doc)return;
 if(e.key==="Escape"&&!$("croppanel").hidden){closeCropPanel();return}
 if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();keyJob(doUndo);return}
 if(e.ctrlKey||e.metaKey||e.altKey)return;
 var k=e.key,sh=e.shiftKey;
 if(["Enter","Tab","j","k","ArrowDown","ArrowUp","t","T","e","E"].indexOf(k)>=0)e.preventDefault();
 /* keys are queued and read the current spot only when they run, so fast typing never hits the same spot twice */
 keyJob(function(){return handleKey(k,sh)})});
async function handleKey(k,sh){var L=curLine();
 if(k==="?"){$("help").classList.add("on");return}
 if(k==="z"||k==="Z"){await doUndo();return}
 if(!L)return;
 if(k==="j"||k==="ArrowDown"){await selectLine(S.cur.line+1);return}
 if(k==="k"||k==="ArrowUp"){await selectLine(S.cur.line-1);return}
 if(k==="Enter"){await acceptLine(false);return}
 if(k==="e"||k==="E"){openTb("line",{});return}
 if(k==="Tab"){var n=nextSpot(L,S.cur.mark,sh?-1:1);if(n!==null){S.cur.mark=n;render();setCursor()}return}
 var s=spots(L).find(function(x){return x.id===S.cur.mark});
 if(!s)return;
 if(s.t==="brk"){if(k==="1"||k==="2"){var b=document.querySelector('#brkbox [data-w="'+(k==="1"?"one":"two")+'"]');if(b){b.click();await chain}}return}
 if(/^[1-9]$/.test(k)){var i=+k-1;if(i<s.alts.length)await chooseSpot(s.id,{k:"alt",i:i});return}
 if(k==="t"||k==="T"){openTb("choose",{mark:s.id});return}
 if(k==="u"||k==="U"){await chooseSpot(s.id,{k:"unreadable"});return}
 if(k==="n"||k==="N"){await chooseSpot(s.id,{k:"empty"});return}}
setInterval(function(){call("GET","/api/ping").then(function(){if(S.reconnected){S.reconnected=false;if(S.page)loadPage(S.page)}}).catch(function(){S.reconnected=true})},6000);
layout();
(async function(){await loadPages();var want=+store("ta_page")||0;var p=S.pages.find(function(x){return x.page===want&&!x.error})||S.pages.find(function(x){return !x.error});
 if(p)await loadPage(p.page);else toast("No page could be opened. "+(S.pages[0]&&S.pages[0].error||""),"err")})();
</script></body></html>
"""
