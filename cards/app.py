"""Local web interface for the yellow-card model.

    .venv/bin/python cards/app.py     then open http://127.0.0.1:5001

The model is fitted ONCE at startup (about 20 seconds) and held in memory, so
every prediction after that is instant. The command-line tool refits on each
invocation, which is fine for one fixture and painful for a matchday slate.

Runs on localhost only. Nothing is sent anywhere.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features
import loader
import predict as predict_mod

app = Flask(__name__)
PREDICTOR: predict_mod.CardPredictor | None = None
TEAMS: dict[str, list[str]] = {}
REFEREES: list[str] = []


def pretty_ref(canon: str) -> str:
    """"m oliver" -> "M Oliver". Canonical form is lowercase because it is an
    identity key, not a label; the UI should not show the plumbing."""
    parts = canon.split(" ", 1)
    if len(parts) == 2:
        return f"{parts[0].upper()} {parts[1].title()}"
    return canon.title()


def boot():
    global PREDICTOR, TEAMS, REFEREES
    print("fitting model, this takes about 20 seconds ...", flush=True)
    PREDICTOR = predict_mod.CardPredictor().fit(verbose=True)
    TEAMS = {d: PREDICTOR.teams(d) for d in sorted(loader.DIVISIONS)}
    REFEREES = PREDICTOR.referees()
    print(f"ready: {sum(len(v) for v in TEAMS.values())} team entries, "
          f"{len(REFEREES)} referees", flush=True)


PAGE = """<!doctype html>
<title>Yellow Card Model</title>
<style>
  :root{
    --bg:#f6f7f9; --panel:#fff; --ink:#15181d; --muted:#666e7a;
    --line:#e2e6eb; --accent:#1c6fd6; --good:#0d7a3d; --bad:#a3352b;
    --bar:#c9d8ef; --bar2:#1c6fd6;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme=light]){
    --bg:#14171c; --panel:#1c2026; --ink:#e8ebef; --muted:#97a0ac;
    --line:#2c323a; --accent:#5c9df5; --good:#4cc47f; --bad:#e8796b;
    --bar:#2f3a4a; --bar2:#5c9df5;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding-block:28px;padding-left:20px;padding-right:20px}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--muted);margin:0 0 22px;font-size:13px}
  .grid{display:grid;grid-template-columns:360px 1fr;gap:20px;align-items:start}
  @media(max-width:820px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:18px}
  label{display:block;font-size:12px;color:var(--muted);margin:12px 0 4px;
    text-transform:uppercase;letter-spacing:.04em}
  select,input{width:100%;padding:8px 10px;border:1px solid var(--line);
    border-radius:6px;background:var(--bg);color:var(--ink);font-size:14px}
  .row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  button{margin-top:18px;width:100%;padding:11px;border:0;border-radius:6px;
    background:var(--accent);color:#fff;font-size:15px;font-weight:600;
    cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th,td{padding:5px 8px;text-align:right;border-bottom:1px solid var(--line)}
  th{font-size:11px;color:var(--muted);text-transform:uppercase;font-weight:600;
    letter-spacing:.04em}
  td:first-child,th:first-child{text-align:left}
  .bar{height:14px;background:var(--bar);border-radius:3px;display:block}
  .bar.hi{background:var(--bar2)}
  .meta{color:var(--muted);font-size:13px;margin:0 0 14px}
  .meta b{color:var(--ink);font-weight:600}
  .ev{margin-top:14px;padding:11px 13px;border-radius:7px;font-size:14px;
    border:1px solid var(--line)}
  .ev.value{border-color:var(--good);color:var(--good)}
  .ev.pass{color:var(--muted)}
  .why{margin-top:16px;font-size:13px;color:var(--muted)}
  .why div{display:flex;justify-content:space-between;padding:2px 0}
  .err{color:var(--bad);white-space:pre-wrap;font-family:ui-monospace,monospace;
    font-size:12px}
  .note{font-size:12px;color:var(--muted);margin-top:16px;line-height:1.5}
  .placeholder{color:var(--muted);text-align:center;padding:50px 0}
</style>
<div class="wrap">
  <h1>Yellow card model</h1>
  <p class="sub">English tiers 1&ndash;4 &middot; negative binomial, 44 features
    &middot; trained on 50,624 matches</p>
  <div class="grid">
    <div class="card">
      <label>Division</label>
      <select id="div"></select>
      <label>Home team</label><select id="home"></select>
      <label>Away team</label><select id="away"></select>
      <label>Date</label><input type="date" id="date">
      <label>Referee (optional)</label><select id="ref"></select>

      <label>Match odds &mdash; all three or none</label>
      <div class="row">
        <input id="oh" placeholder="home" inputmode="decimal">
        <input id="od" placeholder="draw" inputmode="decimal">
        <input id="oa" placeholder="away" inputmode="decimal">
      </div>

      <label>Your card price (optional)</label>
      <div class="row2">
        <input id="line" value="3.5" inputmode="decimal">
        <input id="price" placeholder="odds for over" inputmode="decimal">
      </div>

      <button id="go">Predict</button>
    </div>

    <div class="card" id="out">
      <div class="placeholder">Pick a fixture and press Predict.</div>
    </div>
  </div>
  <p class="note">
    Calibration out of sample: accurate to 0.004 at the 3.5 line. The model has
    never been compared against a real card price, so a positive EV here means
    &ldquo;the arithmetic works&rdquo;, not &ldquo;place this bet&rdquo;.
  </p>
</div>
<script>
let TEAMS={}, REFS=[];
const $=id=>document.getElementById(id);
function opts(sel, items, blank){
  sel.innerHTML = (blank?`<option value="">${blank}</option>`:"") +
    items.map(t => typeof t === "string"
      ? `<option>${t}</option>`
      : `<option value="${t.id}">${t.label}</option>`).join("");
}
fetch("/meta").then(r=>r.json()).then(d=>{
  TEAMS=d.teams; REFS=d.referees;
  opts($("div"), Object.keys(d.divisions).map(k=>k+" — "+d.divisions[k]));
  $("date").value=d.today;
  opts($("ref"), REFS, "not known");
  fillTeams();
});
$("div").onchange=fillTeams;
function divCode(){ return $("div").value.split(" ")[0]; }
function fillTeams(){
  const t=TEAMS[divCode()]||[];
  opts($("home"),t); opts($("away"),t);
  if(t.length>1) $("away").selectedIndex=1;
}
$("go").onclick=async()=>{
  $("go").disabled=true; $("go").textContent="working…";
  const body={division:divCode(),home:$("home").value,away:$("away").value,
    date:$("date").value,referee:$("ref").value||null,
    odds_home:$("oh").value,odds_draw:$("od").value,odds_away:$("oa").value,
    line:$("line").value,price:$("price").value};
  try{
    const r=await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const d=await r.json();
    $("out").innerHTML = d.error ? `<div class="err">${d.error}</div>` : render(d);
  }catch(e){ $("out").innerHTML=`<div class="err">${e}</div>`; }
  $("go").disabled=false; $("go").textContent="Predict";
};
function render(d){
  const mx=Math.max(...d.pmf);
  let rows=d.pmf.map((p,k)=>`<tr>
     <td>${k}${k===9?"+":""}</td>
     <td>${(p*100).toFixed(1)}%</td>
     <td style="width:38%"><span class="bar ${k===d.mode?'hi':''}"
        style="width:${(p/mx*100).toFixed(1)}%"></span></td>
     <td>${(k+0.5).toFixed(1)}</td>
     <td>${(d.over[k]*100).toFixed(1)}%</td></tr>`).join("");
  let ev="";
  if(d.ev){
    const cls=d.ev.edge>0?"value":"pass";
    ev=`<div class="ev ${cls}">over ${d.ev.line} @ ${d.ev.price} &middot;
      implied ${(d.ev.implied*100).toFixed(1)}% &middot;
      model ${(d.ev.model*100).toFixed(1)}% &middot;
      <b>EV ${(d.ev.edge*100>=0?"+":"")}${(d.ev.edge*100).toFixed(1)}%</b>
      ${d.ev.edge>0?" &mdash; value":" &mdash; pass"}</div>`;
  }
  const why=d.why.map(w=>`<div><span>${w.feature}</span>
      <span>${w.pct>=0?"+":""}${(w.pct*100).toFixed(1)}%</span></div>`).join("");
  return `<p class="meta"><b>${d.home}</b> v <b>${d.away}</b> &middot;
      ${d.division_name} &middot; ${d.date}<br>${d.referee_line}<br>
      ${d.market_line}<br>
      League baseline <b>${d.baseline.toFixed(2)}</b> &rarr;
      model expectation <b>${d.mu.toFixed(2)}</b> yellows</p>
    <table><thead><tr><th>yellows</th><th>exactly</th><th></th>
      <th>line</th><th>over</th></tr></thead><tbody>${rows}</tbody></table>
    ${ev}
    <div class="why"><b>What moved it from the league baseline</b>${why}</div>`;
}
</script>
"""


@app.get("/")
def index():
    return PAGE


@app.get("/meta")
def meta():
    import datetime as dt
    return jsonify(teams=TEAMS,
                   referees=[{"id": r, "label": pretty_ref(r)} for r in REFEREES],
                   divisions=loader.DIVISIONS,
                   today=dt.date.today().isoformat())


def _f(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


@app.post("/predict")
def do_predict():
    try:
        b = request.get_json(force=True)
        odds = {}
        oh, od, oa = _f(b.get("odds_home")), _f(b.get("odds_draw")), _f(b.get("odds_away"))
        if oh and od and oa:
            odds.update(home=oh, draw=od, away=oa)

        res = PREDICTOR.predict(b["home"], b["away"], b["division"],
                                b["date"], b.get("referee") or None,
                                odds or None)
        pmf = list(res["pmf"][:9]) + [float(1.0 - res["cdf"][8])]
        over = [float(1.0 - res["cdf"][k]) for k in range(10)]

        ref_line = "Referee: not supplied &mdash; using league average"
        if res["ref_known"] and res["ref_n_career"] >= features.MIN_REF_MATCHES:
            r10 = res["row"].ref_yel10.iloc[0]
            ref_line = (f"Referee: {pretty_ref(res['referee'])} &middot; "
                        f"{res['ref_n_career']} matches on record"
                        + (f" &middot; {r10:.2f}× league average"
                           if np.isfinite(r10) else ""))
        elif res["referee"]:
            ref_line = (f"Referee: {pretty_ref(res['referee'])} &middot; too few matches "
                        "on record, using league average")

        if res.get("mkt_known"):
            ph = (1.0 + res["mkt_home_edge"] - res["mkt_p_draw"]) / 2.0
            pa = ph - res["mkt_home_edge"]
            mkt_line = (f"Market: home {ph:.0%} / draw {res['mkt_p_draw']:.0%} "
                        f"/ away {pa:.0%} &middot; closeness "
                        f"{res['mkt_closeness']:.2f}")
        else:
            mkt_line = ("Market: no odds given &mdash; running without the "
                        "market features")

        con = PREDICTOR.contributions(res["row"])
        why = [{"feature": r.feature, "pct": float(np.expm1(r.contribution))}
               for _, r in con.iterrows()]

        ev = None
        line, price = _f(b.get("line")), _f(b.get("price"))
        if line is not None and price:
            k = int(np.floor(line))
            p = float(1.0 - res["cdf"][k]) if k < len(res["cdf"]) else 0.0
            ev = {"line": line, "price": price, "model": p,
                  "implied": 1.0 / price, "edge": p * price - 1.0}

        return jsonify(home=res["home"], away=res["away"], date=res["date"],
                       division_name=loader.DIVISIONS[res["division"]],
                       referee_line=ref_line, market_line=mkt_line,
                       baseline=res["baseline"], mu=res["mu"],
                       pmf=pmf, over=over, mode=int(np.argmax(pmf)),
                       why=why, ev=ev)
    except Exception:
        return jsonify(error=traceback.format_exc(limit=3)), 200


if __name__ == "__main__":
    boot()
    app.run(host="127.0.0.1", port=5001, debug=False)
