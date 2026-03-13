#!/usr/bin/env python3
"""
HPC Multi-Agent Scheduler System
==================================
PLACER-only scheduler (Rule T, power primary + affinity slack).
No RA rearrangement agent.

Uses a local LLM via Ollama REST API.  Both agents output JSON.
Results written to 4 CSV files: decisions, placements, step_stats, llm_stats.
"""

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests as _req


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MachineState:
    machine_id: int
    cpu_cap: float; io_cap: float; nw_cap: float; mem_cap: float
    p_idle: float;  p_max:  float
    cpu_used: float = 0.0; io_used:  float = 0.0
    nw_used:  float = 0.0; mem_used: float = 0.0

    @property
    def cpu_util(self) -> float:
        return self.cpu_used / self.cpu_cap if self.cpu_cap > 0 else 0.0

    def _load_factor(self, cu, iu, nu, mu) -> float:
        uc = cu / self.cpu_cap if self.cpu_cap > 0 else 0.0
        ui = iu / self.io_cap  if self.io_cap  > 0 else 0.0
        un = nu / self.nw_cap  if self.nw_cap  > 0 else 0.0
        um = mu / self.mem_cap if self.mem_cap > 0 else 0.0
        return min(1.0, 0.70*uc**3 + 0.10*ui + 0.05*un + 0.15*um)

    def calc_power(self, d: float) -> float:
        return self.p_idle + d * (self.p_max - self.p_idle)

    def current_power(self) -> float:
        return self.calc_power(self._load_factor(
            self.cpu_used, self.io_used, self.nw_used, self.mem_used))

    def delta_power(self, cpu=0., io=0., nw=0., mem=0.) -> float:
        db = self._load_factor(self.cpu_used, self.io_used, self.nw_used, self.mem_used)
        da = self._load_factor(self.cpu_used+cpu, self.io_used+io,
                               self.nw_used+nw,  self.mem_used+mem)
        return self.calc_power(da) - self.calc_power(db)

    def can_fit(self, cpu, io, nw, mem) -> bool:
        return (self.cpu_used+cpu <= self.cpu_cap and self.io_used+io <= self.io_cap and
                self.nw_used+nw  <= self.nw_cap  and self.mem_used+mem <= self.mem_cap)

    def place_app(self, cpu, io, nw, mem):
        self.cpu_used+=cpu; self.io_used+=io; self.nw_used+=nw; self.mem_used+=mem

    def remove_app(self, cpu, io, nw, mem):
        self.cpu_used=max(0.,self.cpu_used-cpu); self.io_used=max(0.,self.io_used-io)
        self.nw_used =max(0.,self.nw_used-nw);  self.mem_used=max(0.,self.mem_used-mem)


@dataclass
class AppSpec:
    app_id: int
    cpu: float; io: float; nw: float; mem: float
    instances_total: int
    instances_placed: int = 0

    @property
    def instances_left(self) -> int: return self.instances_total - self.instances_placed
    @property
    def is_complete(self) -> bool:   return self.instances_placed >= self.instances_total
    def place_instance(self):         self.instances_placed += 1


@dataclass
class Placement:
    instance_id: int; app_id: int; machine_id: int
    affinity: int = 0; delta_power: float = 0.0; cpu_util_after: float = 0.0


@dataclass
class Swap:
    action: str                  # "Move" | "Swap"
    instance_id:     int = -1;  app_id:      int = -1
    from_machine_id: int = -1;  to_machine_id: int = -1
    instance_id2:    int = -1;  app_id2:     int = -1
    machine2:        int = -1   # original machine of instance2 (for Swap)
    delta_power_cluster: float = 0.0
    delta_affinity: int = 0
    score_improvement: float = 0.0


@dataclass
class SchedulerState:
    machines: List[MachineState]
    apps:     List[AppSpec]
    affinity: np.ndarray    # (num_apps, num_machines)
    forbidden: np.ndarray   # (num_apps, num_machines)
    timestep: int = 0
    _next_iid: int = 0
    history:               List[Placement] = field(default_factory=list)
    rearrangement_history: List[Swap]      = field(default_factory=list)

    def get_eligible_apps(self) -> List[AppSpec]:
        return [a for a in self.apps if not a.is_complete]

    def is_feasible(self, app_id: int, machine_id: int) -> bool:
        if app_id >= len(self.apps) or machine_id >= len(self.machines): return False
        if self.forbidden[app_id, machine_id] == 1: return False
        a = self.apps[app_id]
        return self.machines[machine_id].can_fit(a.cpu, a.io, a.nw, a.mem)

    def get_placed_instances(self) -> List[Placement]:
        """Current snapshot with swaps applied (returns copies)."""
        import copy
        locs = [copy.copy(p) for p in self.history]
        loc_by_iid = {p.instance_id: p for p in locs}
        for sw in self.rearrangement_history:
            if sw.action == "Move":
                p = loc_by_iid.get(sw.instance_id)
                if p: p.machine_id = sw.to_machine_id
            elif sw.action == "Swap":
                p1 = loc_by_iid.get(sw.instance_id)
                p2 = loc_by_iid.get(sw.instance_id2)
                if p1 and p2:
                    p1.machine_id, p2.machine_id = sw.to_machine_id, sw.from_machine_id
        return locs

    def execute_placement(self, app_id: int, machine_id: int) -> Optional[Placement]:
        if not self.is_feasible(app_id, machine_id): return None
        app = self.apps[app_id]; m = self.machines[machine_id]
        dp = m.delta_power(app.cpu, app.io, app.nw, app.mem)
        m.place_app(app.cpu, app.io, app.nw, app.mem)
        app.place_instance()
        p = Placement(instance_id=self._next_iid, app_id=app_id, machine_id=machine_id,
                      affinity=int(self.affinity[app_id, machine_id]),
                      delta_power=dp, cpu_util_after=m.cpu_util)
        self._next_iid += 1
        self.history.append(p)
        self.timestep += 1
        return p

    def execute_swap(self, sw: Swap) -> bool:
        if sw.action == "Move":
            app = self.apps[sw.app_id]
            if self.forbidden[sw.app_id, sw.to_machine_id]: return False
            fm = self.machines[sw.from_machine_id]; tm = self.machines[sw.to_machine_id]
            if not tm.can_fit(app.cpu, app.io, app.nw, app.mem): return False
            fm.remove_app(app.cpu, app.io, app.nw, app.mem)
            tm.place_app(app.cpu, app.io, app.nw, app.mem)
            self.rearrangement_history.append(sw); return True
        elif sw.action == "Swap":
            a1 = self.apps[sw.app_id]; a2 = self.apps[sw.app_id2]
            m1 = self.machines[sw.from_machine_id]; m2 = self.machines[sw.machine2]
            if self.forbidden[sw.app_id2, sw.from_machine_id]: return False
            if self.forbidden[sw.app_id,  sw.machine2]:        return False
            m1.remove_app(a1.cpu, a1.io, a1.nw, a1.mem)
            m2.remove_app(a2.cpu, a2.io, a2.nw, a2.mem)
            ok = (m1.can_fit(a2.cpu,a2.io,a2.nw,a2.mem) and
                  m2.can_fit(a1.cpu,a1.io,a1.nw,a1.mem))
            if not ok:
                m1.place_app(a1.cpu,a1.io,a1.nw,a1.mem)
                m2.place_app(a2.cpu,a2.io,a2.nw,a2.mem)
                return False
            m1.place_app(a2.cpu,a2.io,a2.nw,a2.mem)
            m2.place_app(a1.cpu,a1.io,a1.nw,a1.mem)
            self.rearrangement_history.append(sw); return True
        return False

    def get_scratchpad(self, n: int = 10) -> str:
        if not self.history: return "(No placements yet)"
        lines = [f"t={self.timestep}  placed={len(self.history)}  swaps={len(self.rearrangement_history)}"]
        for p in self.history[-n:]:
            lines.append(f"  inst_{p.instance_id}: app_{p.app_id}→m{p.machine_id} "
                         f"(aff={p.affinity} ΔP={p.delta_power:.4f}W)")
        return "\n".join(lines)


# ============================================================================
# Table Formatters
# ============================================================================

def format_remaining(eligible: List[AppSpec]) -> str:
    lines = ["app_id | cpu  | io   | nw   | mem  | left", "-"*46]
    for a in eligible[:30]:
        lines.append(f"{a.app_id:6} | {a.cpu:4.1f} | {a.io:4.1f} | "
                     f"{a.nw:4.1f} | {a.mem:4.1f} | {a.instances_left}")
    if len(eligible) > 30: lines.append(f"...({len(eligible)-30} more)")
    return "\n".join(lines)


def format_candidate_table(state: SchedulerState,
                           eligible: List[AppSpec],
                           max_rows: int = 30) -> str:
    rows = []
    for app in eligible:
        for m in state.machines:
            if not state.is_feasible(app.app_id, m.machine_id): continue
            dp  = m.delta_power(app.cpu, app.io, app.nw, app.mem)
            aff = int(state.affinity[app.app_id, m.machine_id])
            rows.append((dp, app.app_id, m.machine_id, aff))
    if not rows: return "(No feasible candidates)"
    rows.sort(key=lambda r: (r[0], -r[3]))
    rows = rows[:max_rows]
    lines = [f"{'rank':>4} | {'app_id':>6} | {'machine_id':>10} | "
             f"{'delta_total':>11} | {'affinity':>8}", "-"*50]
    for rank, (dp, aid, mid, aff) in enumerate(rows, 1):
        lines.append(f"{rank:>4} | {aid:>6} | {mid:>10} | {dp:>11.6f} | {aff:>8}")
    return "\n".join(lines)


def format_ops_table(state: SchedulerState, max_ops: int = 20) -> str:
    """Pre-computed Pareto-improving Move + Swap operations."""
    placed = state.get_placed_instances()
    if not placed: return "(No placed instances)"
    rows = []

    # ── Moves ────────────────────────────────────────────────────────────────
    for p in placed:
        app = state.apps[p.app_id]; cm = state.machines[p.machine_id]
        cur_aff = int(state.affinity[p.app_id, p.machine_id])
        p_cm_before = cm.current_power()
        for tmid, tm in enumerate(state.machines):
            if tmid == p.machine_id: continue
            if state.forbidden[p.app_id, tmid]: continue
            if not tm.can_fit(app.cpu, app.io, app.nw, app.mem): continue
            to_aff = int(state.affinity[p.app_id, tmid])
            d_cm_a = cm._load_factor(max(0.,cm.cpu_used-app.cpu), max(0.,cm.io_used-app.io),
                                     max(0.,cm.nw_used-app.nw),   max(0.,cm.mem_used-app.mem))
            dpc = (cm.calc_power(d_cm_a) + tm.calc_power(tm._load_factor(
                    tm.cpu_used+app.cpu, tm.io_used+app.io,
                    tm.nw_used+app.nw,  tm.mem_used+app.mem))
                  ) - (p_cm_before + tm.current_power())
            da = to_aff - cur_aff
            if (dpc < 0 and da >= 0) or (dpc <= 0 and da > 0):
                rows.append({"op":"Move","iid":p.instance_id,"aid":p.app_id,
                             "fm":p.machine_id,"tm":tmid,
                             "iid2":"","aid2":"","m2":"","dpc":dpc,"da":da})

    # ── Swaps ────────────────────────────────────────────────────────────────
    for i, p1 in enumerate(placed):
        a1=state.apps[p1.app_id]; m1=state.machines[p1.machine_id]
        aff1c=int(state.affinity[p1.app_id,p1.machine_id])
        pm1b=m1.current_power()
        for p2 in placed[i+1:]:
            if p2.machine_id==p1.machine_id: continue
            a2=state.apps[p2.app_id]; m2=state.machines[p2.machine_id]
            aff2c=int(state.affinity[p2.app_id,p2.machine_id])
            if state.forbidden[p1.app_id,p2.machine_id]: continue
            if state.forbidden[p2.app_id,p1.machine_id]: continue
            # m1 after: -a1 +a2
            m1cu=m1.cpu_used-a1.cpu+a2.cpu; m1iu=m1.io_used-a1.io+a2.io
            m1nu=m1.nw_used-a1.nw+a2.nw;   m1mu=m1.mem_used-a1.mem+a2.mem
            if m1cu>m1.cpu_cap or m1iu>m1.io_cap or m1nu>m1.nw_cap or m1mu>m1.mem_cap: continue
            # m2 after: -a2 +a1
            m2cu=m2.cpu_used-a2.cpu+a1.cpu; m2iu=m2.io_used-a2.io+a1.io
            m2nu=m2.nw_used-a2.nw+a1.nw;   m2mu=m2.mem_used-a2.mem+a1.mem
            if m2cu>m2.cpu_cap or m2iu>m2.io_cap or m2nu>m2.nw_cap or m2mu>m2.mem_cap: continue
            pm2b=m2.current_power()
            pm1a=m1.calc_power(m1._load_factor(m1cu,m1iu,m1nu,m1mu))
            pm2a=m2.calc_power(m2._load_factor(m2cu,m2iu,m2nu,m2mu))
            dpc=(pm1a+pm2a)-(pm1b+pm2b)
            aff1n=int(state.affinity[p1.app_id,p2.machine_id])
            aff2n=int(state.affinity[p2.app_id,p1.machine_id])
            da=(aff1n+aff2n)-(aff1c+aff2c)
            if (dpc < 0 and da >= 0) or (dpc <= 0 and da > 0):
                rows.append({"op":"Swap","iid":p1.instance_id,"aid":p1.app_id,
                             "fm":p1.machine_id,"tm":p2.machine_id,
                             "iid2":p2.instance_id,"aid2":p2.app_id,"m2":p2.machine_id,
                             "dpc":dpc,"da":da})

    if not rows: return "(No Pareto-improving operations found)"
    rows.sort(key=lambda r: (r["dpc"], -r["da"]))
    rows = rows[:max_ops]
    lines=[f"{'rank':>4}|{'op':>4}|{'iid':>4}|{'aid':>4}|{'fm':>5}|{'tm':>5}|"
           f"{'iid2':>5}|{'aid2':>5}|{'m2':>5}|{'dP_cluster':>11}|{'dAff':>5}","-"*74]
    for rank,r in enumerate(rows,1):
        lines.append(f"{rank:>4}|{r['op']:>4}|{r['iid']:>4}|{r['aid']:>4}|"
                     f"{r['fm']:>5}|{r['tm']:>5}|{str(r['iid2']):>5}|"
                     f"{str(r['aid2']):>5}|{str(r['m2']):>5}|"
                     f"{r['dpc']:>11.4f}|{r['da']:>5}")
    return "\n".join(lines)


# ============================================================================
# Prompt Templates
# ============================================================================

PLACER_PROMPT = r"""You are an expert power-aware cluster scheduler.

At this step, you must choose exactly ONE action:
- Place ONE instance of an app (app_id) onto ONE machine (machine_id), OR
- Delay (only if no feasible placement exists), OR
- Stop (only if all instances are scheduled).

HARD CONSTRAINTS (strict, must satisfy now):
1) Anti-affinity: forbidden[app_id][machine_id] must be 0
2) Capacity after placement must satisfy:
   cpu_used + cpu_req <= cpu_cap
   io_used  + io_req  <= io_cap
   bw_used  + bw_req  <= bw_cap
   mem_used + mem_req <= mem_cap

SOFT GOALS (consider BOTH; do NOT combine into a single scalar score):
- Minimize incremental cluster power delta_total (lower is better)
- Maximize affinity payoff A in [0 or 1] (1 is better)

POWER MODEL (already computed in table; shown for understanding only):
  d = min(1, 0.70*u_cpu^3 + 0.10*u_io + 0.05*u_nw + 0.15*u_mem)
  P(m) = P_idle + d*(P_max - P_idle)
  delta_total = P(m)_after - P(m)_before

DECISION POLICY — Rule T (power-first with reasonable affinity preference):
- All rows in the candidate table are already feasible.
- Let DPmin = smallest delta_total among all candidates.
- epsilon = max({eps_rel} * |DPmin|, {eps_min})
- If any candidate has affinity=1 AND delta_total <= DPmin + epsilon:
    choose the affinity=1 candidate with smallest delta_total
  else:
    choose the candidate with the smallest delta_total

IMPORTANT: Output ONLY a JSON object — no markdown, no explanation, no code fences.

Output schemas:
{{"action":"Place","app_id":int,"machine_id":int,"dp_min":float,"epsilon":float,"chosen_dp":float,"chosen_affinity":int,"rationale":"..."}}
{{"action":"Delay","rationale":"..."}}
{{"action":"Stop","rationale":"..."}}

Scratchpad:
{scratchpad}

Remaining instances:
{remaining_instances}

Candidate table (all feasible, sorted by delta_total ascending):
{candidate_table}
"""

REARRANGER_PROMPT = r"""You are an expert post-placement optimizer (Rearrangement Agent).

After the last placement, you may improve the current schedule by ONE operation.

Allowed operations:
1) Move(instance_id, app_id, from_machine, to_machine)
2) Swap(instance_id1, app_id1, machine1, instance_id2, app_id2, machine2)
3) NoChange

HARD CONSTRAINTS: The operations table is already feasibility-filtered.
You MUST NOT invent operations not in the table.

POWER MODEL (shown for understanding; values are pre-computed in the table):
  d = min(1, 0.70*u_cpu^3 + 0.10*u_io + 0.05*u_nw + 0.15*u_mem)
  P(m) = P_idle + d*(P_max - P_idle)
  dP_cluster = net change in total cluster power

IMPROVEMENT RULE — Pareto only (no scalar objective):
Accept ONLY if:
  1) dP_cluster < 0 AND dAff >= 0   (power drops, affinity doesn't worsen)
  OR
  2) dP_cluster <= 0 AND dAff > 0   (power same or better, affinity improves)

Selection: most negative dP_cluster first; tie-break by highest dAff; then first row.

ABSOLUTE RULE: If every row has dP_cluster > 0, output NoChange.

IMPORTANT: Output ONLY a JSON object — no markdown, no explanation, no code fences.

Output schemas:
{{"action":"Move","instance_id":int,"app_id":int,"from_machine":int,"to_machine":int,"delta_power_cluster":float,"delta_affinity":int,"rationale":"..."}}
{{"action":"Swap","instance_id1":int,"app_id1":int,"machine1":int,"instance_id2":int,"app_id2":int,"machine2":int,"delta_power_cluster":float,"delta_affinity":int,"rationale":"..."}}
{{"action":"NoChange","rationale":"..."}}

Scratchpad:
{scratchpad}

Neighborhood operations table (feasible, Pareto-improving only):
{ops_table}
"""


# ============================================================================
# LLM Agent — Ollama REST API
# ============================================================================

@dataclass
class LLMResult:
    text: str; input_tokens: int; output_tokens: int
    elapsed_s: float; model: str


class OllamaAgent:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.url   = f"{base_url}/api/generate"

    def call(self, prompt: str, temperature: float = 0.0) -> LLMResult:
        t0 = time.time()
        resp = _req.post(self.url, json={
            "model": self.model, "prompt": prompt,
            "stream": False, "options": {"temperature": temperature},
        }, timeout=300)
        resp.raise_for_status()
        d = resp.json()
        return LLMResult(
            text=d.get("response",""),
            input_tokens=d.get("prompt_eval_count",0),
            output_tokens=d.get("eval_count",0),
            elapsed_s=time.time()-t0,
            model=self.model,
        )


# ============================================================================
# JSON Parser
# ============================================================================

def _parse_json(result: LLMResult, agent: str, verbose: bool) -> Optional[Dict]:
    if verbose:
        print(f"\n--- {agent} ---\n{result.text}\n---\n")
    text = re.sub(r'```(?:json)?', '', result.text).strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        print(f"[{agent}] No JSON found.")
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError as e:
        print(f"[{agent}] JSON parse error: {e}")
        return None


# ============================================================================
# CSV helpers
# ============================================================================

DECISIONS_COLS = [
    "t","agent","action",
    "instance_id","app_id","machine_id",
    "from_machine","to_machine",
    "instance_id1","machine1","app_id1",
    "instance_id2","machine2","app_id2",
    "dp_min","epsilon","chosen_dp","chosen_affinity",
    "delta_power_cluster","delta_affinity",
    "rationale",
    "llm_seconds","llm_model","llm_input_tokens",
    "llm_output_tokens","llm_total_tokens",
]
PLACEMENTS_COLS = ["instance_id","app_id","machine_id"]
STEP_STATS_COLS = ["t","num_placed","total_power","total_affinity",
                   "llm_calls_total","llm_time_total"]
LLM_STATS_COLS  = ["llm_calls_total","llm_calls_placer","llm_calls_rearranger",
                   "llm_time_total","llm_time_placer","llm_time_rearranger"]

def _open_csv(path, cols):
    f = open(path,"w",newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    return f, w


# ============================================================================
# Orchestrator
# ============================================================================

class HPCSchedulerOrchestrator:
    def __init__(self, state: SchedulerState, model: str = "llama3",
                 ollama_url: str = "http://localhost:11434",
                 max_iterations: int = 1000,
                 eps_rel: float = 0.0, eps_min: float = 0.25,
                 verbose: bool = True, output_dir: str = "."):
        self.state          = state
        self.max_iterations = max_iterations
        self.eps_rel        = eps_rel
        self.eps_min        = eps_min
        self.verbose        = verbose
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_placer   = OllamaAgent(model, ollama_url)
        self.agent_ra       = OllamaAgent(model, ollama_url)
        self._calls_placer  = 0; self._time_placer  = 0.0
        self._calls_ra      = 0; self._time_ra      = 0.0

    def log(self, msg):
        if self.verbose: print(f"[t={self.state.timestep}] {msg}")

    # ── prompts ───────────────────────────────────────────────────────────────

    def _placer_prompt(self) -> str:
        el = self.state.get_eligible_apps()
        return PLACER_PROMPT.format(
            eps_rel=self.eps_rel, eps_min=self.eps_min,
            scratchpad=self.state.get_scratchpad(),
            remaining_instances=format_remaining(el),
            candidate_table=format_candidate_table(self.state, el),
        )

    def _ra_prompt(self) -> str:
        return REARRANGER_PROMPT.format(
            scratchpad=self.state.get_scratchpad(),
            ops_table=format_ops_table(self.state),
        )

    # ── fallback ──────────────────────────────────────────────────────────────

    def _fallback(self) -> Optional[Tuple[int,int,float]]:
        best = None
        for app in self.state.get_eligible_apps():
            for m in self.state.machines:
                if not self.state.is_feasible(app.app_id, m.machine_id): continue
                dp = m.delta_power(app.cpu, app.io, app.nw, app.mem)
                if best is None or dp < best[2]:
                    best = (app.app_id, m.machine_id, dp)
        return best

    # ── single iteration ──────────────────────────────────────────────────────

    def run_iteration(self, dec_w, step_w) -> bool:
        t  = self.state.timestep
        el = self.state.get_eligible_apps()
        if not el:
            self.log("No eligible apps — stopping."); return False

        # ── PLACER ────────────────────────────────────────────────────────────
        self.log("Calling PLACER...")
        res = self.agent_placer.call(self._placer_prompt())
        self._calls_placer += 1; self._time_placer += res.elapsed_s
        data = _parse_json(res, "PLACER", self.verbose)

        placed: Optional[Placement] = None
        dr = {k:"" for k in DECISIONS_COLS}
        dr.update({"t":t,"agent":"placer",
                   "llm_seconds":res.elapsed_s,"llm_model":res.model,
                   "llm_input_tokens":res.input_tokens,
                   "llm_output_tokens":res.output_tokens,
                   "llm_total_tokens":res.input_tokens+res.output_tokens})

        if data and data.get("action") == "Stop":
            dr["action"]="Stop"; dr["rationale"]=data.get("rationale","")
            dec_w.writerow(dr); return False

        if data and data.get("action") == "Place":
            placed = self.state.execute_placement(int(data["app_id"]), int(data["machine_id"]))
            if placed:
                dr.update({"action":"Place","instance_id":placed.instance_id,
                           "app_id":placed.app_id,"machine_id":placed.machine_id,
                           "dp_min":data.get("dp_min",""),
                           "epsilon":data.get("epsilon",""),
                           "chosen_dp":data.get("chosen_dp",""),
                           "chosen_affinity":data.get("chosen_affinity",""),
                           "rationale":data.get("rationale","")})
            else:
                self.log("PLACER proposed infeasible placement — fallback.")

        if placed is None:
            fb = self._fallback()
            if fb is None: self.log("No feasible placement — stopping."); return False
            placed = self.state.execute_placement(fb[0], fb[1])
            dr.update({"action":"Place","instance_id":placed.instance_id,
                       "app_id":placed.app_id,"machine_id":placed.machine_id,
                       "chosen_dp":fb[2],"chosen_affinity":placed.affinity,
                       "rationale":"deterministic fallback"})

        self.log(f"Placed inst_{placed.instance_id}: app_{placed.app_id}→"
                 f"m{placed.machine_id} (aff={placed.affinity} ΔP={placed.delta_power:.4f}W)")
        dec_w.writerow(dr)

        # ── RA disabled in this variant ──────────────────────────────────────
        rr = {k:"" for k in DECISIONS_COLS}
        rr.update({"t":t,"agent":"rearranger","action":"NoChange",
                   "rationale":"RA disabled in this variant",
                   "llm_seconds":0,"llm_model":self.agent_ra.model,
                   "llm_input_tokens":0,"llm_output_tokens":0,"llm_total_tokens":0})
        dec_w.writerow(rr)

        # ── step stats ────────────────────────────────────────────────────────
        step_w.writerow({
            "t":t, "num_placed":len(self.state.history),
            "total_power":sum(m.current_power() for m in self.state.machines),
            "total_affinity":sum(p.affinity for p in self.state.history),
            "llm_calls_total":self._calls_placer+self._calls_ra,
            "llm_time_total":self._time_placer+self._time_ra,
        })
        return True

    # ── full run ──────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        self.log("="*70)
        self.log("HPC Scheduler — PLACER only, no RA (Power primary, Affinity secondary)")
        self.log("="*70)

        dec_f,  dec_w  = _open_csv(self.output_dir/"decisions.csv",  DECISIONS_COLS)
        step_f, step_w = _open_csv(self.output_dir/"step_stats.csv", STEP_STATS_COLS)

        try:
            for _ in range(self.max_iterations):
                if not self.run_iteration(dec_w, step_w): break
        finally:
            dec_f.flush();  dec_f.close()
            step_f.flush(); step_f.close()

        # placements.csv
        with open(self.output_dir/"placements.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=PLACEMENTS_COLS); w.writeheader()
            for p in self.state.get_placed_instances():
                w.writerow({"instance_id":p.instance_id,"app_id":p.app_id,
                            "machine_id":p.machine_id})

        # llm_stats.csv
        tc = self._calls_placer+self._calls_ra
        tt = self._time_placer +self._time_ra
        with open(self.output_dir/"llm_stats.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=LLM_STATS_COLS); w.writeheader()
            w.writerow({"llm_calls_total":tc,"llm_calls_placer":self._calls_placer,
                        "llm_calls_rearranger":self._calls_ra,
                        "llm_time_total":tt,"llm_time_placer":self._time_placer,
                        "llm_time_rearranger":self._time_ra})

        tp = sum(m.current_power() for m in self.state.machines)
        ta = sum(p.affinity for p in self.state.history)
        results = {"total_placements":len(self.state.history),
                   "total_swaps":len(self.state.rearrangement_history),
                   "total_power_W":tp,"total_affinity":ta,
                   "llm_calls":tc,"llm_time_s":tt,
                   "output_dir":str(self.output_dir)}

        self.log("="*70)
        self.log(f"Done. Placements={results['total_placements']}  Swaps={results['total_swaps']}")
        self.log(f"Power={tp:.2f}W  Affinity={ta}  LLM calls={tc}  Time={tt:.1f}s")
        self.log(f"CSVs → {self.output_dir}/")
        self.log("="*70)
        return results


# ============================================================================
# Data Loading
# ============================================================================

def load_data_from_csv(apps_file, machines_file, affinity_file, forbidden_file):
    apps_df = pd.read_csv(apps_file)
    apps = [AppSpec(app_id=i,cpu=r["CPU"],io=r["IO"],nw=r["NW"],mem=r["Memory"],
                    instances_total=int(r["Instances"]))
            for i,r in apps_df.iterrows()]
    machines_df = pd.read_csv(machines_file)
    machines = [MachineState(machine_id=i,cpu_cap=r["CPU"],io_cap=r["IO"],
                             nw_cap=r["NW"],mem_cap=r["Memory"],
                             p_idle=r["P_idle"],p_max=r["P_max"])
                for i,r in machines_df.iterrows()]
    affinity  = pd.read_csv(affinity_file,  index_col=0).values.astype(int)
    forbidden = pd.read_csv(forbidden_file, index_col=0).values.astype(int)
    return SchedulerState(machines=machines,apps=apps,
                          affinity=affinity,forbidden=forbidden)


def generate_random_state(num_apps, num_machines, seed=None,
                          affinity_density=0.3, forbidden_density=0.1,
                          instances_range=(1,4)):
    rng = np.random.default_rng(seed)
    machines = [MachineState(machine_id=i,
                             cpu_cap=float(rng.integers(100,251)),
                             io_cap=float(rng.integers(100,301)),
                             nw_cap=float(rng.integers(100,301)),
                             mem_cap=float(rng.integers(128,513)),
                             p_idle=float(rng.integers(60,111)),
                             p_max=float(rng.integers(140,271)))
                for i in range(num_machines)]
    apps = [AppSpec(app_id=i,cpu=float(rng.integers(5,61)),
                    io=float(rng.integers(5,41)),nw=float(rng.integers(5,41)),
                    mem=float(rng.integers(5,61)),
                    instances_total=int(rng.integers(instances_range[0],instances_range[1]+1)))
            for i in range(num_apps)]
    affinity  = (rng.random((num_apps,num_machines)) < affinity_density).astype(int)
    forbidden = (rng.random((num_apps,num_machines)) < forbidden_density).astype(int)
    for i in range(num_apps):
        if forbidden[i].all(): forbidden[i, rng.integers(num_machines)] = 0
    return SchedulerState(machines=machines,apps=apps,
                          affinity=affinity,forbidden=forbidden)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="HPC Multi-Agent Scheduler (PLACER + RA)")
    p.add_argument("--model",        default="llama3")
    p.add_argument("--ollama-url",   default="http://localhost:11434")
    p.add_argument("--max-iter",     type=int,   default=1000)
    p.add_argument("--eps-rel",      type=float, default=0.0)
    p.add_argument("--eps-min",      type=float, default=0.25)
    p.add_argument("--output-dir",   default="results")
    p.add_argument("--verbose",      action="store_true")
    # CSV files
    p.add_argument("--apps",         default="Applications.csv")
    p.add_argument("--machines",     default="Machines.csv")
    p.add_argument("--affinity",     default="Affinity.csv")
    p.add_argument("--forbidden",    default="AntiAffinity.csv")
    # Subset size — works for BOTH csv and synthetic modes
    p.add_argument("--num-apps",     type=int, default=None,
                   help="Use first N apps from CSV (or generate N apps if --synthetic)")
    p.add_argument("--num-machines", type=int, default=None,
                   help="Use first M machines from CSV (or generate M machines if --synthetic)")
    # Synthetic mode
    p.add_argument("--synthetic",    action="store_true",
                   help="Generate random data instead of loading CSVs")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    if args.synthetic:
        num_apps     = args.num_apps     or 10
        num_machines = args.num_machines or 20
        state = generate_random_state(num_apps, num_machines, seed=args.seed)
        print(f"Synthetic mode: {num_apps} apps × {num_machines} machines (seed={args.seed})")

    else:
        # Load CSVs, then optionally slice to first N apps / M machines
        apps_df     = pd.read_csv(args.apps)
        machines_df = pd.read_csv(args.machines)
        affinity_df = pd.read_csv(args.affinity,  index_col=0)
        forbidden_df= pd.read_csv(args.forbidden,  index_col=0)

        num_apps     = args.num_apps     or len(apps_df)
        num_machines = args.num_machines or len(machines_df)

        # Clamp to available rows
        num_apps     = min(num_apps,     len(apps_df))
        num_machines = min(num_machines, len(machines_df))

        apps_df      = apps_df.iloc[:num_apps]
        machines_df  = machines_df.iloc[:num_machines]
        affinity     = affinity_df.iloc[:num_apps, :num_machines].values.astype(int)
        forbidden    = forbidden_df.iloc[:num_apps, :num_machines].values.astype(int)

        apps = [AppSpec(app_id=i, cpu=r["CPU"], io=r["IO"], nw=r["NW"],
                        mem=r["Memory"], instances_total=int(r["Instances"]))
                for i, r in enumerate(apps_df.to_dict("records"))]

        machines = [MachineState(machine_id=i, cpu_cap=r["CPU"], io_cap=r["IO"],
                                 nw_cap=r["NW"], mem_cap=r["Memory"],
                                 p_idle=r["P_idle"], p_max=r["P_max"])
                    for i, r in enumerate(machines_df.to_dict("records"))]

        state = SchedulerState(machines=machines, apps=apps,
                               affinity=affinity, forbidden=forbidden)

        total_inst = sum(a.instances_total for a in apps)
        print(f"CSV subset: {num_apps} apps × {num_machines} machines "
              f"({total_inst} total instances)")

    orch = HPCSchedulerOrchestrator(
        state=state, model=args.model, ollama_url=args.ollama_url,
        max_iterations=args.max_iter, eps_rel=args.eps_rel, eps_min=args.eps_min,
        verbose=args.verbose, output_dir=args.output_dir,
    )
    orch.run()
