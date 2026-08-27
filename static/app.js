const $ = s => document.querySelector(s);
const STEP_ORDER = ["retrieve","draft","verify","reflect","html_gen","html_revise","git"];
const SSE_RETRY_DELAYS_MS = [500, 1000, 2000];
let RUN = null, TOKEN = "", streamAbort = null;

function api(path, method, body){
  return fetch(path, {method,
    headers:{"Authorization":"Bearer "+TOKEN, "Content-Type":"application/json"},
    body: body ? JSON.stringify(body) : undefined});
}
function show(id){ $(id).classList.remove("hidden"); }
function hide(id){ $(id).classList.add("hidden"); }
function clearChildren(el){ while(el.firstChild) el.removeChild(el.firstChild); }

function initSteps(){
  const root = $("#steps");
  clearChildren(root);
  for(const n of STEP_ORDER){
    const el = document.createElement("div");
    el.className = "step"; el.id = "step-"+n;
    const dot = document.createElement("span"); dot.className = "dot";
    const name = document.createElement("span"); name.className = "name";
    name.textContent = n.replace("_"," ");
    const meta = document.createElement("span"); meta.className = "meta";
    el.append(dot, name, meta);
    if(n==="html_revise") el.style.opacity = ".45";
    root.appendChild(el);
  }
}
function setRunning(on){
  const sub = $("#pipeSub");
  clearChildren(sub);
  if(on){
    const spin = document.createElement("span"); spin.className = "spin";
    const t = document.createElement("span"); t.textContent = " running…";
    sub.append(spin, t);
  }
}
function finishNode(node, headline){
  const el = $("#step-"+node); if(!el) return;
  el.classList.remove("active"); el.classList.add("done");
  const bits=[];
  if("web_sources" in headline) bits.push(headline.web_sources+" web");
  if("kb_results" in headline) bits.push(headline.kb_results+" kb");
  if("claims" in headline) bits.push(headline.claims+" claims");
  if("grounding_score" in headline) bits.push("g="+headline.grounding_score);
  if("reflection_score" in headline) bits.push("refl "+headline.reflection_score+"/10");
  if("iterations" in headline) bits.push("iter "+headline.iterations);
  if("git_status" in headline) bits.push(headline.git_status);
  if("latency_ms" in headline) bits.push(Math.round(headline.latency_ms)+"ms");
  const m = el.querySelector(".meta"); m.textContent = bits.join(" · ");
}

function authFailed(){
  TOKEN = "";
  const field = $("#token");
  if(field) field.value = "";
  show("#setup");
  onError("authentication failed");
}

function parseSseBlock(block){
  const data = [];
  for(const line of block.split("\n")){
    if(line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if(!data.length) return null;
  return JSON.parse(data.join("\n"));
}

async function consumeSse(res){
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  let sawEnd = false;
  while(true){
    const {value, done} = await reader.read();
    if(done) break;
    buf += value;
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for(const block of parts){
      if(!block.trim() || block.startsWith(":")) continue;
      const d = parseSseBlock(block);
      if(!d) continue;
      if(d.event==="node") finishNode(d.node, d.headline||{});
      else if(d.event==="gate") onGate(d.review);
      else if(d.event==="done") onDone(d);
      else if(d.event==="error") onError(d.error);
      else if(d.event==="segment_end"){ sawEnd = true; return true; }
    }
  }
  if(buf.trim() && !buf.startsWith(":")){
    const d = parseSseBlock(buf);
    if(d && d.event==="segment_end") return true;
    if(d && d.event==="done"){ onDone(d); return true; }
  }
  return sawEnd;
}

async function openStream(attempt){
  attempt = attempt || 0;
  if(streamAbort) streamAbort.abort();
  streamAbort = new AbortController();
  setRunning(true);
  let res;
  try{
    res = await fetch(`/ui/runs/${RUN}/events`, {
      headers: {"Authorization":"Bearer "+TOKEN},
      signal: streamAbort.signal,
    });
  }catch(err){
    if(err && err.name === "AbortError") return;
    return retryStream(attempt);
  }
  if(res.status === 401 || res.status === 403){
    authFailed();
    return;
  }
  if(!res.ok){
    return retryStream(attempt);
  }
  let ended;
  try{
    ended = await consumeSse(res);
  }catch(err){
    if(err && err.name === "AbortError") return;
    ended = false;
  }
  if(ended){
    setRunning(false);
    return;
  }
  return retryStream(attempt);
}

async function retryStream(attempt){
  if(attempt >= SSE_RETRY_DELAYS_MS.length){
    onError("stream ended unexpectedly");
    return;
  }
  try{
    const st = await api(`/runs/${RUN}`, "GET");
    if(st.status === 401 || st.status === 403){ authFailed(); return; }
    if(st.ok){
      const body = await st.json();
      if(body.status === "awaiting_review" && body.review){
        onGate(body.review);
        return;
      }
      if(body.status === "complete" || body.status === "rejected"){
        onDone({status: body.status, summary: body.summary || {}});
        return;
      }
    }
  }catch(_e){ /* fall through to retry */ }
  await new Promise(r => setTimeout(r, SSE_RETRY_DELAYS_MS[attempt]));
  return openStream(attempt + 1);
}

function onGate(r){
  setRunning(false);
  $("#pipeSub").textContent = "waiting for your review";
  if(r.type==="hitl_review"){
    const rows = r.review_claims || [];
    const calculatedUvr = rows.length
      ? rows.filter(row => String(row.status || "").toLowerCase()==="unverified").length / rows.length
      : null;
    const uvr = typeof r.unverified_rate === "number" ? r.unverified_rate : calculatedUvr;
    const uvrText = typeof uvr === "number" ? `${(uvr * 100).toFixed(1)}%` : "not computable";
    const provenance = r.reflection_provenance || {};
    const reflection = provenance.origin === "judge" && provenance.parse_status === "ok"
      ? `REAL JUDGE SCORE ${r.reflection_score}/10`
      : `FALLBACK / UNAVAILABLE SCORE ${r.reflection_score}/10 (${provenance.reason || "unknown"})`;
    $("#g1meta").textContent =
      `Claim grounding ${(r.grounding_score??0).toFixed(2)} · UVR ${uvrText} of the extracted verdict set (not a completeness measure) · ${reflection} — ${r.reflection_notes||""}`;
    $("#g1frame").srcdoc = r.draft_review_html || "";
    renderGrounding(rows);
    $("#g1fb").value=""; show("#gate1"); $("#gate1").scrollIntoView({behavior:"smooth"});
  } else if(r.type==="hitl_html_review"){
    $("#g2meta").textContent = `${r.html_filename||""} · grounding ${(r.grounding_score??0).toFixed(2)}`;
    $("#g2frame").srcdoc = r.html_output || "";
    const w = r.validation_warnings||[];
    if(w.length){ $("#g2warn").textContent = "warnings: "+w.join("; "); show("#g2warn"); } else hide("#g2warn");
    $("#g2fb").value=""; show("#gate2"); $("#gate2").scrollIntoView({behavior:"smooth"});
  }
}

function renderGrounding(report){
  const root = $("#g1grounding");
  clearChildren(root);
  const h = document.createElement("h3"); h.textContent = "Claims & sources";
  root.appendChild(h);
  if(!report.length){
    const p = document.createElement("p"); p.className = "muted"; p.textContent = "No claims extracted.";
    root.appendChild(p);
    return;
  }
  const counts = {total: report.length, verified: 0, weak: 0, unverified: 0};
  for(const row of report){
    const status = String(row.status || "").toLowerCase();
    if(Object.prototype.hasOwnProperty.call(counts, status)) counts[status] += 1;
  }
  const summary = document.createElement("div");
  summary.id = "g1claimSummary";
  summary.className = "claim-summary";
  for(const [label, value] of [
    ["Total", counts.total], ["Verified", counts.verified],
    ["Weak", counts.weak], ["Unverified", counts.unverified],
  ]){
    const stat = document.createElement("span");
    stat.className = "claim-count "+label.toLowerCase();
    stat.textContent = `${label}: ${value}`;
    summary.appendChild(stat);
  }
  root.appendChild(summary);

  const filters = document.createElement("div");
  filters.className = "claim-filters";
  filters.setAttribute("role", "group");
  filters.setAttribute("aria-label", "Filter claims by verification status");
  root.appendChild(filters);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for(const label of ["Claim","Status","Conf.","Source"]){
    const th = document.createElement("th"); th.textContent = label; hr.appendChild(th);
  }
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = document.createElement("tbody");
  for(const c of report){
    const tr = document.createElement("tr");
    const st = String(c.status||"").toLowerCase();
    tr.className = "claim-row status-"+(st || "unknown");
    const claim = document.createElement("td"); claim.textContent = c.claim || "";
    const stTd = document.createElement("td");
    const tag = document.createElement("span");
    tag.className = "tag "+st; tag.textContent = st || "?";
    stTd.appendChild(tag);
    const conf = document.createElement("td"); conf.textContent = (c.confidence??0).toFixed(2);
    const src = document.createElement("td");
    let sourceUrl = null;
    try{
      const parsed = new URL(c.source_url || "");
      if(parsed.protocol === "https:" && parsed.hostname) sourceUrl = parsed.href;
    }catch(_e){ /* server already validates; browser still treats malformed data as text */ }
    if(sourceUrl){
      const a = document.createElement("a");
      a.href = sourceUrl; a.target = "_blank"; a.rel = "noopener noreferrer nofollow";
      a.textContent = c.source_label || "source";
      src.appendChild(a);
    } else {
      const span = document.createElement("span"); span.className = "muted";
      span.textContent = c.source_label || "none";
      src.appendChild(span);
    }
    tr.append(claim, stTd, conf, src);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  root.appendChild(table);

  const applyFilter = (filter) => {
    for(const row of tbody.querySelectorAll("tr.claim-row")){
      const matches = filter === "all" || row.classList.contains("status-"+filter);
      row.classList.toggle("hidden", !matches);
    }
    for(const button of filters.querySelectorAll("button")){
      button.setAttribute("aria-pressed", String(button.dataset.filter === filter));
    }
  };
  for(const filter of ["all", "verified", "weak", "unverified"]){
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary claim-filter";
    button.dataset.filter = filter;
    button.textContent = filter.toUpperCase();
    button.addEventListener("click", () => applyFilter(filter));
    filters.appendChild(button);
  }
  applyFilter("all");
}

function setGateButtons(disabled){ document.querySelectorAll("[data-act]").forEach(b=>b.disabled=disabled); }
async function gateAction(gate, act){
  const fb = gate==="1" ? $("#g1fb").value.trim() : $("#g2fb").value.trim();
  if(act==="feedback" && !fb){ alert("Enter the change you want before requesting changes."); return; }
  setGateButtons(true);
  let path = `/ui/runs/${RUN}/${act}`, body = null;
  if(act==="feedback"){ body = {feedback: fb}; }
  const res = await api(path, "POST", body);
  if(res.status === 401 || res.status === 403){ authFailed(); setGateButtons(false); return; }
  if(!res.ok){ const t=await res.text(); onError(`${res.status} ${t}`); setGateButtons(false); return; }
  hide("#gate1"); hide("#gate2"); setGateButtons(false);
  $("#pipeSub").textContent = act==="approve" ? "approved — continuing…"
      : act==="feedback" ? "revising…" : "rejecting…";
  openStream();
}
document.querySelectorAll("[data-act]").forEach(b=>{
  b.addEventListener("click", ()=>gateAction(b.dataset.gate, b.dataset.act));
});

function onDone(d){
  if(streamAbort){ streamAbort.abort(); streamAbort=null; }
  hide("#gate1"); hide("#gate2");
  const s=d.summary||{}; const base=$("#siteBase").value.trim().replace(/\/+$/,"");
  const publishable = d.status==="complete" && (s.git_status==="merged" || s.git_status==="tagged_and_merged");
  const body = $("#doneBody");
  clearChildren(body);
  const p = document.createElement("p");
  if(publishable){
    p.className = "ok";
    p.textContent = "Merged locally (git_status: "+(s.git_status||"?")+"). Click Publish to push the approved commit.";
  } else if(d.status==="complete"){
    p.className = "ok";
    p.textContent = "Published locally (git_status: "+(s.git_status||"?")+", file "+(s.html_filename||"?")+").";
    if(s.slug && base){
      const url = `${base}/${s.slug}.html`;
      const extra = document.createElement("p");
      const a = document.createElement("a"); a.href = url; a.target="_blank"; a.rel="noopener noreferrer"; a.textContent = url;
      extra.appendChild(a); body.appendChild(p); body.appendChild(extra);
    }
  } else if(d.status==="rejected"){
    p.className = "bad"; p.textContent = "Run ended: nothing was published.";
  }
  if(!body.contains(p)) body.appendChild(p);
  const meta = document.createElement("p"); meta.className = "muted";
  meta.textContent = `grounding ${s.grounding_score??"?"} · cost $${(s.total_cost_usd??0)}`;
  body.appendChild(meta);
  clearChildren($("#livePanel"));
  $("#publishBtn").disabled = false;
  if(publishable) show("#publishBtns"); else hide("#publishBtns");
  $("#pipeSub").textContent="finished"; show("#done"); $("#done").scrollIntoView({behavior:"smooth"});
  $("#go").disabled=false;
}

async function publishRun(){
  $("#publishBtn").disabled = true;
  const res = await api(`/ui/runs/${RUN}/publish`, "POST");
  const panel = $("#livePanel");
  clearChildren(panel);
  if(!res.ok){
    const t = await res.text();
    const p = document.createElement("p"); p.className = "bad";
    p.textContent = `publish failed: ${res.status} ${t}`;
    panel.appendChild(p);
    $("#publishBtn").disabled = false;
    return;
  }
  const j = await res.json();
  const p = document.createElement("p"); p.className = "ok";
  p.style.fontSize = "16px"; p.style.fontWeight = "700";
  p.textContent = "LIVE — ";
  const a = document.createElement("a");
  a.href = j.live_url; a.target="_blank"; a.rel="noopener noreferrer"; a.textContent = j.live_url;
  p.appendChild(a);
  panel.appendChild(p);
  hide("#publishBtns");
}
$("#publishBtn").addEventListener("click", publishRun);
$("#g2expand").addEventListener("click", ()=>{
  $("#g2frame").classList.toggle("expanded");
});

document.addEventListener("keydown", (e)=>{
  const tag = (e.target && e.target.tagName) || "";
  if(tag==="TEXTAREA" || tag==="INPUT") return;
  const key = e.key.toLowerCase();
  const gate1Open = !$("#gate1").classList.contains("hidden");
  const gate2Open = !$("#gate2").classList.contains("hidden");
  if(gate1Open || gate2Open){
    const gate = gate1Open ? "1" : "2";
    if(key==="a"){ e.preventDefault(); gateAction(gate, "approve"); }
    else if(key==="r"){ e.preventDefault(); gateAction(gate, "reject"); }
    else if(key==="c"){ e.preventDefault(); $(gate==="1" ? "#g1fb" : "#g2fb").focus(); }
  }
  if(key==="p" && !$("#publishBtns").classList.contains("hidden")){
    e.preventDefault(); publishRun();
  }
});
function onError(msg){
  if(streamAbort){ streamAbort.abort(); streamAbort=null; }
  const sub = $("#pipeSub");
  clearChildren(sub);
  const span = document.createElement("span"); span.className = "bad";
  span.textContent = "error: "+msg;
  sub.appendChild(span);
  $("#go").disabled=false; setGateButtons(false);
}

$("#go").addEventListener("click", async ()=>{
  const topic=$("#topic").value.trim();
  const field = $("#token");
  TOKEN = field.value || TOKEN;
  field.value = "";
  if(!topic||!TOKEN){ alert("Topic and token are required."); return; }
  $("#go").disabled=true; hide("#gate1"); hide("#gate2"); hide("#done");
  initSteps(); show("#pipeline"); $("#pipeSub").textContent="running…";
  const res=await api("/ui/runs","POST",{topic, series:$("#series").value.trim()||"Learning Log"});
  if(res.status === 401 || res.status === 403){ authFailed(); return; }
  if(!res.ok){ onError(`${res.status} ${await res.text()}`); return; }
  const j=await res.json(); RUN=j.run_id;
  $("#runPill").textContent="run "+RUN.slice(0,8); openStream();
});
