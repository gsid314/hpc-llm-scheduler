#!/usr/bin/env python3
"""
HPC Multi-Agent Debate Scheduler System
=========================================
Four-agent system for VM/container placement with configurable debate:
  - Agent POWER_ADVOCATE   : Argues for power-minimising placements
  - Agent AFFINITY_ADVOCATE : Argues for affinity-maximising placements
  - Agent PLACEMENT_JUDGE  : Reads debate transcript, makes final placement
  - Agent RA               : Post-placement Pareto-improving rearrangement

Debate rounds are configurable:
  debate_rounds=0  → no debate, placement agent decides alone
  debate_rounds=3  → 3 rounds of debate before placement
  debate_rounds=5  → 5 rounds of debate before placement

ALL LLM conversations are recorded in conversations.csv.
Uses a local LLM via Ollama REST API.
"""

import csv
import copy
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
# Data Structures (unchanged from original)
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
    action: str
    instance_id:     int = -1;  app_id:      int = -1
    from_machine_id: int = -1;  to_machine_id: int = -1
    instance_id2:    int = -1;  app_id2:     int = -1
    machine2:        int = -1
    delta_power_cluster: float = 0.0
    delta_affinity: int = 0
    score_improvement: float = 0.0


@dataclass
class SchedulerState:
    machines: List[MachineState]
    apps:     List[AppSpec]
    affinity: np.ndarray
    forbidden: np.ndarray
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
            lines.append(f"  inst_{p.instance_id}: app_{p.app_id}->m{p.machine_id} "
                         f"(aff={p.affinity} dP={p.delta_power:.4f}W)")
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
    placed = state.get_placed_instances()
    if not placed: return "(No placed instances)"
    rows = []

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
            m1cu=m1.cpu_used-a1.cpu+a2.cpu; m1iu=m1.io_used-a1.io+a2.io
            m1nu=m1.nw_used-a1.nw+a2.nw;   m1mu=m1.mem_used-a1.mem+a2.mem
            if m1cu>m1.cpu_cap or m1iu>m1.io_cap or m1nu>m1.nw_cap or m1mu>m1.mem_cap: continue
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
# Prompt Templates — 4 Agents
# ============================================================================

# --- Power Advocate: argues purely for minimum delta_power ---
# {num_choices_instruction} is filled dynamically based on num_choices config
POWER_ADVOCATE_PROMPT = r"""You are the POWER ADVOCATE agent in a multi-agent HPC scheduler.
Your role: argue for the placement(s) that MINIMISE incremental cluster power (delta_total).

POWER MODEL (cubic):
  d = min(1, 0.70*u_cpu^3 + 0.10*u_io + 0.05*u_nw + 0.15*u_mem)
  P(m) = P_idle + d*(P_max - P_idle)
  delta_total = P(m)_after - P(m)_before

Candidate table (all feasible, sorted by delta_total ascending):
{candidate_table}

Scratchpad:
{scratchpad}

{debate_context}

{num_choices_instruction}

Explain WHY your recommended machine(s) are most power-efficient (consider cubic CPU model, consolidation).
If another agent proposed a different placement, counter their argument from a power perspective.

Output ONLY a JSON object — no markdown, no code fences:
{output_schema}
"""

# --- Affinity Advocate: argues purely for maximum affinity ---
# {num_choices_instruction} is filled dynamically based on num_choices config
AFFINITY_ADVOCATE_PROMPT = r"""You are the AFFINITY ADVOCATE agent in a multi-agent HPC scheduler.
Your role: argue for the placement(s) that MAXIMISE affinity (affinity=1 placements).

Affinity rules:
- affinity[app_id][machine_id] = 1 means the app has preference for that machine
- More affinity=1 placements = better co-location, reduced network hops, better SLA

Candidate table (all feasible, sorted by delta_total ascending):
{candidate_table}

Remaining instances:
{remaining_instances}

Scratchpad:
{scratchpad}

{debate_context}

{num_choices_instruction}

If no affinity=1 candidate exists, acknowledge this and suggest the best available.
If another agent proposed a different placement, counter their argument from an affinity perspective.

Output ONLY a JSON object — no markdown, no code fences:
{output_schema}
"""

# ============================================================================
# num_choices helpers — dynamic prompt fragments
# ============================================================================

def _power_choices_instruction(num_choices: int) -> str:
    """Return instruction text for Power Advocate based on num_choices."""
    if num_choices == 1:
        return ("Recommend exactly 1 placement — the single candidate with the LOWEST "
                "delta_total. Rank: 1.")
    elif num_choices <= 0:  # 0 means full rank
        return ("Rank ALL candidates from the table by delta_total (ascending). "
                "Provide a full ranked list with your power argument for each.")
    else:
        return (f"Recommend your TOP {num_choices} placements ranked by delta_total "
                f"(ascending). Provide your power argument for each.")


def _affinity_choices_instruction(num_choices: int) -> str:
    """Return instruction text for Affinity Advocate based on num_choices."""
    if num_choices == 1:
        return ("Recommend exactly 1 placement — the best affinity=1 candidate "
                "(tie-break by lowest delta_total). Rank: 1.")
    elif num_choices <= 0:
        return ("Rank ALL candidates from the table by affinity (descending), "
                "then by delta_total (ascending). Provide a full ranked list "
                "with your affinity argument for each.")
    else:
        return (f"Recommend your TOP {num_choices} placements ranked by affinity "
                f"(descending), then delta_total (ascending). "
                f"Provide your affinity argument for each.")


def _power_output_schema(num_choices: int) -> str:
    if num_choices == 1:
        return ('{{"recommendations":[{{"rank":1,"app_id":int,"machine_id":int,'
                '"delta_total":float}}],"argument":"your detailed power argument..."}}')
    elif num_choices <= 0:
        return ('{{"recommendations":[{{"rank":int,"app_id":int,"machine_id":int,'
                '"delta_total":float}},...(all candidates)],'
                '"argument":"your detailed power argument..."}}')
    else:
        return ('{{"recommendations":[{{"rank":int,"app_id":int,"machine_id":int,'
                f'"delta_total":float}},...(up to {num_choices})],'
                '"argument":"your detailed power argument..."}}')


def _affinity_output_schema(num_choices: int) -> str:
    if num_choices == 1:
        return ('{{"recommendations":[{{"rank":1,"app_id":int,"machine_id":int,'
                '"affinity":int,"delta_total":float}}],'
                '"argument":"your detailed affinity argument..."}}')
    elif num_choices <= 0:
        return ('{{"recommendations":[{{"rank":int,"app_id":int,"machine_id":int,'
                '"affinity":int,"delta_total":float}},...(all candidates)],'
                '"argument":"your detailed affinity argument..."}}')
    else:
        return ('{{"recommendations":[{{"rank":int,"app_id":int,"machine_id":int,'
                f'"affinity":int,"delta_total":float}},...(up to {num_choices})],'
                '"argument":"your detailed affinity argument..."}}')


def _format_debate_recommendations(agent_name: str, data: Optional[Dict],
                                   raw_text: str) -> str:
    """Format debate agent recommendations into a readable transcript block."""
    if not data:
        return f"  [{agent_name}] (parse error) Raw: {raw_text[:300]}"

    recs = data.get("recommendations", [])
    arg  = data.get("argument", "")

    if not recs:
        # Fallback: old single-recommendation format
        app = data.get("recommended_app_id", data.get("app_id", "?"))
        mid = data.get("recommended_machine_id", data.get("machine_id", "?"))
        dp  = data.get("delta_total", "?")
        aff = data.get("affinity", "?")
        return (f"  [{agent_name}] Recommends: app_id={app}, machine_id={mid}, "
                f"delta_total={dp}, affinity={aff}\n  Argument: {arg}")

    lines = [f"  [{agent_name}] {len(recs)} recommendation(s):"]
    for r in recs:
        lines.append(f"    Rank {r.get('rank','?')}: app_id={r.get('app_id','?')}, "
                      f"machine_id={r.get('machine_id','?')}, "
                      f"delta_total={r.get('delta_total','?')}, "
                      f"affinity={r.get('affinity','?')}")
    lines.append(f"  Argument: {arg}")
    return "\n".join(lines)


# --- Placement Judge: reads debate, makes final decision ---
PLACEMENT_JUDGE_PROMPT = r"""You are the PLACEMENT JUDGE in a multi-agent HPC scheduler.
You must choose exactly ONE action: Place, Delay, or Stop.

HARD CONSTRAINTS (must satisfy):
1) Anti-affinity: forbidden[app_id][machine_id] must be 0
2) Capacity: cpu_used + cpu_req <= cpu_cap (and same for io, nw, mem)

SOFT GOALS (consider BOTH, do NOT combine into a single scalar):
- Minimize incremental cluster power delta_total (lower is better)
- Maximize affinity payoff A in [0 or 1] (1 is better)

POWER MODEL:
  d = min(1, 0.70*u_cpu^3 + 0.10*u_io + 0.05*u_nw + 0.15*u_mem)
  P(m) = P_idle + d*(P_max - P_idle)

DECISION POLICY — Rule T (power-first with reasonable affinity preference):
- Let DPmin = smallest delta_total among all candidates.
- epsilon = max({eps_rel} * |DPmin|, {eps_min})
- If any candidate has affinity=1 AND delta_total <= DPmin + epsilon:
    choose the affinity=1 candidate with smallest delta_total
  else:
    choose the candidate with the smallest delta_total

{debate_section}

Candidate table (all feasible, sorted by delta_total ascending):
{candidate_table}

Remaining instances:
{remaining_instances}

Scratchpad:
{scratchpad}

IMPORTANT: Output ONLY a JSON object — no markdown, no explanation, no code fences.

Output schemas:
{{"action":"Place","app_id":int,"machine_id":int,"dp_min":float,"epsilon":float,"chosen_dp":float,"chosen_affinity":int,"rationale":"..."}}
{{"action":"Delay","rationale":"..."}}
{{"action":"Stop","rationale":"..."}}
"""

# --- RA (rearrangement) — same as original ---
REARRANGER_PROMPT = r"""You are an expert post-placement optimizer (Rearrangement Agent).

After the last placement, you may improve the current schedule by ONE operation.

Allowed operations:
1) Move(instance_id, app_id, from_machine, to_machine)
2) Swap(instance_id1, app_id1, machine1, instance_id2, app_id2, machine2)
3) NoChange

HARD CONSTRAINTS: The operations table is already feasibility-filtered.
You MUST NOT invent operations not in the table.

POWER MODEL:
  d = min(1, 0.70*u_cpu^3 + 0.10*u_io + 0.05*u_nw + 0.15*u_mem)
  P(m) = P_idle + d*(P_max - P_idle)

IMPROVEMENT RULE — Pareto only:
Accept ONLY if:
  1) dP_cluster < 0 AND dAff >= 0
  OR
  2) dP_cluster <= 0 AND dAff > 0

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
LLM_STATS_COLS  = ["llm_calls_total","llm_calls_placer","llm_calls_debate",
                   "llm_calls_rearranger",
                   "llm_time_total","llm_time_placer","llm_time_debate",
                   "llm_time_rearranger",
                   "llm_tokens_total","llm_tokens_placer","llm_tokens_debate",
                   "llm_tokens_rearranger"]

# Conversation log columns — captures EVERY LLM call
CONVERSATION_COLS = [
    "conversation_id",   # unique per placement step
    "timestep",          # scheduler timestep
    "turn_number",       # sequential within this conversation
    "agent_name",        # power_advocate / affinity_advocate / placement_judge / rearranger
    "debate_round",      # 0 for non-debate calls, 1..N for debate rounds
    "prompt",            # full prompt sent to LLM
    "response",          # full LLM response text
    "parsed_json",       # parsed JSON (or empty if parse failed)
    "input_tokens",
    "output_tokens",
    "elapsed_seconds",
    "model",
    "debate_rounds_config",  # the configured debate_rounds for this run
    "num_choices_config",    # 1, 3, or 0 (full rank)
]


def _open_csv(path, cols):
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    return f, w


# ============================================================================
# Orchestrator — with Debate
# ============================================================================

class HPCDebateOrchestrator:
    def __init__(self, state: SchedulerState, model: str = "llama3",
                 ollama_url: str = "http://localhost:11434",
                 max_iterations: int = 1000,
                 eps_rel: float = 0.0, eps_min: float = 0.25,
                 debate_rounds: int = 3,
                 num_choices: int = 1,
                 verbose: bool = True, output_dir: str = "."):
        self.state          = state
        self.max_iterations = max_iterations
        self.eps_rel        = eps_rel
        self.eps_min        = eps_min
        self.debate_rounds  = debate_rounds
        self.num_choices    = num_choices   # 1, 3, or 0 (full rank)
        self.verbose        = verbose
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 4 agents — all use same Ollama model
        self.agent_power    = OllamaAgent(model, ollama_url)
        self.agent_affinity = OllamaAgent(model, ollama_url)
        self.agent_placer   = OllamaAgent(model, ollama_url)
        self.agent_ra       = OllamaAgent(model, ollama_url)

        # Stats
        self._calls_placer  = 0; self._time_placer  = 0.0; self._tokens_placer  = 0
        self._calls_debate  = 0; self._time_debate  = 0.0; self._tokens_debate  = 0
        self._calls_ra      = 0; self._time_ra      = 0.0; self._tokens_ra      = 0

        # Conversation counter
        self._conv_id = 0

    def log(self, msg):
        if self.verbose: print(f"[t={self.state.timestep}] {msg}")

    # ── conversation logger ──────────────────────────────────────────────────

    def _log_conversation(self, conv_w, agent_name: str, debate_round: int,
                          turn_number: int, prompt: str, result: LLMResult,
                          parsed: Optional[Dict]):
        conv_w.writerow({
            "conversation_id":      self._conv_id,
            "timestep":             self.state.timestep,
            "turn_number":          turn_number,
            "agent_name":           agent_name,
            "debate_round":         debate_round,
            "prompt":               prompt,
            "response":             result.text,
            "parsed_json":          json.dumps(parsed) if parsed else "",
            "input_tokens":         result.input_tokens,
            "output_tokens":        result.output_tokens,
            "elapsed_seconds":      f"{result.elapsed_s:.3f}",
            "model":                result.model,
            "debate_rounds_config": self.debate_rounds,
            "num_choices_config":   self.num_choices,
        })

    # ── debate phase ─────────────────────────────────────────────────────────

    def _run_debate(self, conv_w) -> Tuple[str, int]:
        """
        Run Power vs Affinity debate for self.debate_rounds rounds.
        Each advocate outputs self.num_choices recommendations (0=full rank).
        Returns (debate_transcript_str, turn_counter).
        If debate_rounds=0, returns empty transcript.
        """
        if self.debate_rounds == 0:
            return "", 0

        el = self.state.get_eligible_apps()
        candidate_table = format_candidate_table(self.state, el)
        remaining = format_remaining(el)
        scratchpad = self.state.get_scratchpad()

        nc = self.num_choices
        nc_label = "full rank" if nc <= 0 else str(nc)

        transcript_lines = []
        turn = 0

        for rnd in range(1, self.debate_rounds + 1):
            if transcript_lines:
                debate_ctx = "Previous debate rounds:\n" + "\n".join(transcript_lines)
            else:
                debate_ctx = "This is the first round of debate. No prior arguments yet."

            # --- Power Advocate ---
            power_prompt = POWER_ADVOCATE_PROMPT.format(
                candidate_table=candidate_table,
                scratchpad=scratchpad,
                debate_context=debate_ctx,
                num_choices_instruction=_power_choices_instruction(nc),
                output_schema=_power_output_schema(nc),
            )
            turn += 1
            self.log(f"  Debate R{rnd}: Power Advocate ({nc_label} choices)...")
            power_res = self.agent_power.call(power_prompt, temperature=0.3)
            self._calls_debate += 1; self._time_debate += power_res.elapsed_s
            self._tokens_debate += power_res.input_tokens + power_res.output_tokens
            power_data = _parse_json(power_res, f"POWER_R{rnd}", self.verbose)
            self._log_conversation(conv_w, "power_advocate", rnd, turn,
                                   power_prompt, power_res, power_data)

            power_block = _format_debate_recommendations(
                f"Round {rnd} - POWER ADVOCATE", power_data, power_res.text)
            transcript_lines.append(power_block)

            # --- Affinity Advocate ---
            debate_ctx = "Previous debate rounds:\n" + "\n".join(transcript_lines)

            affinity_prompt = AFFINITY_ADVOCATE_PROMPT.format(
                candidate_table=candidate_table,
                remaining_instances=remaining,
                scratchpad=scratchpad,
                debate_context=debate_ctx,
                num_choices_instruction=_affinity_choices_instruction(nc),
                output_schema=_affinity_output_schema(nc),
            )
            turn += 1
            self.log(f"  Debate R{rnd}: Affinity Advocate ({nc_label} choices)...")
            aff_res = self.agent_affinity.call(affinity_prompt, temperature=0.3)
            self._calls_debate += 1; self._time_debate += aff_res.elapsed_s
            self._tokens_debate += aff_res.input_tokens + aff_res.output_tokens
            aff_data = _parse_json(aff_res, f"AFFINITY_R{rnd}", self.verbose)
            self._log_conversation(conv_w, "affinity_advocate", rnd, turn,
                                   affinity_prompt, aff_res, aff_data)

            aff_block = _format_debate_recommendations(
                f"Round {rnd} - AFFINITY ADVOCATE", aff_data, aff_res.text)
            transcript_lines.append(aff_block)

        transcript = "\n\n".join(transcript_lines)
        return transcript, turn

    # ── single iteration ─────────────────────────────────────────────────────

    def run_iteration(self, dec_w, step_w, conv_w) -> bool:
        t  = self.state.timestep
        el = self.state.get_eligible_apps()
        if not el:
            self.log("No eligible apps — stopping."); return False

        self._conv_id += 1

        # ── DEBATE PHASE ─────────────────────────────────────────────────────
        self.log(f"=== Debate phase (rounds={self.debate_rounds}) ===")
        debate_transcript, turn = self._run_debate(conv_w)

        # ── PLACEMENT JUDGE ──────────────────────────────────────────────────
        self.log("Calling PLACEMENT JUDGE...")
        el = self.state.get_eligible_apps()  # refresh after potential state reads

        if debate_transcript:
            debate_section = (
                "DEBATE TRANSCRIPT — Power Advocate vs Affinity Advocate:\n"
                "Read both perspectives carefully, then apply Rule T to make your decision.\n\n"
                f"{debate_transcript}\n\n"
                "END OF DEBATE TRANSCRIPT."
            )
        else:
            debate_section = "(No debate was conducted. Decide based on the candidate table and Rule T.)"

        placer_prompt = PLACEMENT_JUDGE_PROMPT.format(
            eps_rel=self.eps_rel, eps_min=self.eps_min,
            debate_section=debate_section,
            candidate_table=format_candidate_table(self.state, el),
            remaining_instances=format_remaining(el),
            scratchpad=self.state.get_scratchpad(),
        )
        turn += 1
        res = self.agent_placer.call(placer_prompt)
        self._calls_placer += 1; self._time_placer += res.elapsed_s
        self._tokens_placer += res.input_tokens + res.output_tokens
        data = _parse_json(res, "PLACEMENT_JUDGE", self.verbose)
        self._log_conversation(conv_w, "placement_judge", 0, turn,
                               placer_prompt, res, data)

        placed: Optional[Placement] = None
        dr = {k:"" for k in DECISIONS_COLS}
        dr.update({"t":t, "agent":"placement_judge",
                   "llm_seconds":res.elapsed_s, "llm_model":res.model,
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
                self.log("PLACEMENT JUDGE proposed infeasible — fallback.")

        if placed is None:
            fb = self._fallback()
            if fb is None: self.log("No feasible placement — stopping."); return False
            placed = self.state.execute_placement(fb[0], fb[1])
            dr.update({"action":"Place","instance_id":placed.instance_id,
                       "app_id":placed.app_id,"machine_id":placed.machine_id,
                       "chosen_dp":fb[2],"chosen_affinity":placed.affinity,
                       "rationale":"deterministic fallback"})

        self.log(f"Placed inst_{placed.instance_id}: app_{placed.app_id}->"
                 f"m{placed.machine_id} (aff={placed.affinity} dP={placed.delta_power:.4f}W)")
        dec_w.writerow(dr)

        # ── RA (Rearrangement Agent) ─────────────────────────────────────────
        self.log("Calling RA...")
        ops = format_ops_table(self.state)
        rr = {k:"" for k in DECISIONS_COLS}
        rr.update({"t":t,"agent":"rearranger"})

        if ops.startswith("(No"):
            self.log("RA: no Pareto-improving ops — skip LLM.")
            rr.update({"action":"NoChange","rationale":"no Pareto-improving ops in table",
                       "llm_seconds":0,"llm_model":self.agent_ra.model,
                       "llm_input_tokens":0,"llm_output_tokens":0,"llm_total_tokens":0})
        else:
            ra_prompt = REARRANGER_PROMPT.format(
                scratchpad=self.state.get_scratchpad(),
                ops_table=ops,
            )
            turn += 1
            rres = self.agent_ra.call(ra_prompt)
            self._calls_ra += 1; self._time_ra += rres.elapsed_s
            self._tokens_ra += rres.input_tokens + rres.output_tokens
            rd = _parse_json(rres, "RA", self.verbose)
            self._log_conversation(conv_w, "rearranger", 0, turn,
                                   ra_prompt, rres, rd)
            rr.update({"llm_seconds":rres.elapsed_s,"llm_model":rres.model,
                       "llm_input_tokens":rres.input_tokens,
                       "llm_output_tokens":rres.output_tokens,
                       "llm_total_tokens":rres.input_tokens+rres.output_tokens})

            if rd and rd.get("action") == "Move":
                sw = Swap(action="Move",
                          instance_id=int(rd["instance_id"]), app_id=int(rd["app_id"]),
                          from_machine_id=int(rd["from_machine"]),
                          to_machine_id=int(rd["to_machine"]),
                          delta_power_cluster=float(rd.get("delta_power_cluster",0)),
                          delta_affinity=int(rd.get("delta_affinity",0)))
                pareto=((sw.delta_power_cluster<0 and sw.delta_affinity>=0) or
                        (sw.delta_power_cluster<=0 and sw.delta_affinity>0))
                ok = pareto and self.state.execute_swap(sw)
                if ok:
                    self.log(f"RA Move: inst_{sw.instance_id} m{sw.from_machine_id}->m{sw.to_machine_id}")
                    rr.update({"action":"Move","instance_id":sw.instance_id,
                               "app_id":sw.app_id,"from_machine":sw.from_machine_id,
                               "to_machine":sw.to_machine_id,
                               "delta_power_cluster":sw.delta_power_cluster,
                               "delta_affinity":sw.delta_affinity,
                               "rationale":rd.get("rationale","")})
                else:
                    rr.update({"action":"NoChange","rationale":"Move failed server-side validation"})

            elif rd and rd.get("action") == "Swap":
                swap_required = ["instance_id1","app_id1","machine1",
                                 "instance_id2","app_id2","machine2"]
                missing = [k for k in swap_required if k not in rd]
                if missing:
                    self.log(f"RA Swap missing fields {missing} -- NoChange.")
                    rr.update({"action":"NoChange",
                               "rationale":f"Swap response missing fields: {missing}"})
                else:
                    sw = Swap(action="Swap",
                              instance_id=int(rd["instance_id1"]), app_id=int(rd["app_id1"]),
                              from_machine_id=int(rd["machine1"]),
                              to_machine_id=int(rd["machine2"]),
                              instance_id2=int(rd["instance_id2"]), app_id2=int(rd["app_id2"]),
                              machine2=int(rd["machine1"]),
                              delta_power_cluster=float(rd.get("delta_power_cluster",0)),
                              delta_affinity=int(rd.get("delta_affinity",0)))
                    pareto=((sw.delta_power_cluster<0 and sw.delta_affinity>=0) or
                            (sw.delta_power_cluster<=0 and sw.delta_affinity>0))
                    ok = pareto and self.state.execute_swap(sw)
                    if ok:
                        self.log(f"RA Swap: inst_{sw.instance_id} <-> inst_{sw.instance_id2}")
                        rr.update({"action":"Swap",
                                   "instance_id1":sw.instance_id,"app_id1":sw.app_id,
                                   "machine1":sw.from_machine_id,
                                   "instance_id2":sw.instance_id2,"app_id2":sw.app_id2,
                                   "machine2":sw.machine2,
                                   "delta_power_cluster":sw.delta_power_cluster,
                                   "delta_affinity":sw.delta_affinity,
                                   "rationale":rd.get("rationale","")})
                    else:
                        rr.update({"action":"NoChange","rationale":"Swap failed server-side validation"})
            else:
                rr["action"]="NoChange"
                rr["rationale"]=rd.get("rationale","") if rd else "parse error"

        dec_w.writerow(rr)

        # ── step stats ────────────────────────────────────────────────────────
        step_w.writerow({
            "t":t, "num_placed":len(self.state.history),
            "total_power":sum(m.current_power() for m in self.state.machines),
            "total_affinity":sum(p.affinity for p in self.state.history),
            "llm_calls_total":self._calls_placer+self._calls_debate+self._calls_ra,
            "llm_time_total":self._time_placer+self._time_debate+self._time_ra,
        })
        return True

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

    # ── full run ──────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        self.log("="*70)
        self.log(f"HPC Debate Scheduler — debate_rounds={self.debate_rounds}  "
                 f"num_choices={self.num_choices}")
        self.log(f"  Agents: POWER_ADVOCATE + AFFINITY_ADVOCATE + PLACEMENT_JUDGE + RA")
        self.log("="*70)

        dec_f,  dec_w  = _open_csv(self.output_dir/"decisions.csv",  DECISIONS_COLS)
        step_f, step_w = _open_csv(self.output_dir/"step_stats.csv", STEP_STATS_COLS)
        conv_f, conv_w = _open_csv(self.output_dir/"conversations.csv", CONVERSATION_COLS)

        try:
            for _ in range(self.max_iterations):
                if not self.run_iteration(dec_w, step_w, conv_w): break
        finally:
            for f in [dec_f, step_f, conv_f]:
                f.flush(); f.close()

        # placements.csv
        with open(self.output_dir/"placements.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=PLACEMENTS_COLS); w.writeheader()
            for p in self.state.get_placed_instances():
                w.writerow({"instance_id":p.instance_id,"app_id":p.app_id,
                            "machine_id":p.machine_id})

        # llm_stats.csv
        tc = self._calls_placer + self._calls_debate + self._calls_ra
        tt = self._time_placer  + self._time_debate  + self._time_ra
        tk = self._tokens_placer + self._tokens_debate + self._tokens_ra
        with open(self.output_dir/"llm_stats.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=LLM_STATS_COLS); w.writeheader()
            w.writerow({
                "llm_calls_total":tc,
                "llm_calls_placer":self._calls_placer,
                "llm_calls_debate":self._calls_debate,
                "llm_calls_rearranger":self._calls_ra,
                "llm_time_total":tt,
                "llm_time_placer":self._time_placer,
                "llm_time_debate":self._time_debate,
                "llm_time_rearranger":self._time_ra,
                "llm_tokens_total":tk,
                "llm_tokens_placer":self._tokens_placer,
                "llm_tokens_debate":self._tokens_debate,
                "llm_tokens_rearranger":self._tokens_ra,
            })

        tp = sum(m.current_power() for m in self.state.machines)
        ta = sum(p.affinity for p in self.state.history)
        results = {
            "debate_rounds": self.debate_rounds,
            "num_choices": self.num_choices,
            "total_placements":len(self.state.history),
            "total_swaps":len(self.state.rearrangement_history),
            "total_power_W":tp, "total_affinity":ta,
            "llm_calls":tc, "llm_calls_debate":self._calls_debate,
            "llm_time_s":tt, "llm_time_debate_s":self._time_debate,
            "llm_tokens_total":tk,
            "llm_tokens_placer":self._tokens_placer,
            "llm_tokens_debate":self._tokens_debate,
            "llm_tokens_ra":self._tokens_ra,
            "output_dir":str(self.output_dir),
        }

        self.log("="*70)
        self.log(f"Done. debate_rounds={self.debate_rounds}  "
                 f"num_choices={self.num_choices}  "
                 f"Placements={results['total_placements']}  "
                 f"Swaps={results['total_swaps']}")
        self.log(f"Power={tp:.2f}W  Affinity={ta}  "
                 f"LLM calls={tc} (debate={self._calls_debate})  Time={tt:.1f}s")
        self.log(f"Tokens total={tk}  placer={self._tokens_placer}  "
                 f"debate={self._tokens_debate}  ra={self._tokens_ra}")
        self.log(f"CSVs -> {self.output_dir}/")
        self.log("="*70)
        return results


# ============================================================================
# Data Loading (reused from original)
# ============================================================================

def load_data_from_csv(apps_file, machines_file, affinity_file, forbidden_file):
    apps_df = pd.read_csv(apps_file)
    apps = [AppSpec(app_id=i, cpu=r["CPU"], io=r["IO"], nw=r["NW"], mem=r["Memory"],
                    instances_total=int(r["Instances"]))
            for i, r in apps_df.iterrows()]
    machines_df = pd.read_csv(machines_file)
    machines = [MachineState(machine_id=i, cpu_cap=r["CPU"], io_cap=r["IO"],
                             nw_cap=r["NW"], mem_cap=r["Memory"],
                             p_idle=r["P_idle"], p_max=r["P_max"])
                for i, r in machines_df.iterrows()]
    affinity  = pd.read_csv(affinity_file,  index_col=0).values.astype(int)
    forbidden = pd.read_csv(forbidden_file, index_col=0).values.astype(int)
    return SchedulerState(machines=machines, apps=apps,
                          affinity=affinity, forbidden=forbidden)


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
    apps = [AppSpec(app_id=i, cpu=float(rng.integers(5,61)),
                    io=float(rng.integers(5,41)), nw=float(rng.integers(5,41)),
                    mem=float(rng.integers(5,61)),
                    instances_total=int(rng.integers(instances_range[0],instances_range[1]+1)))
            for i in range(num_apps)]
    affinity  = (rng.random((num_apps,num_machines)) < affinity_density).astype(int)
    forbidden = (rng.random((num_apps,num_machines)) < forbidden_density).astype(int)
    for i in range(num_apps):
        if forbidden[i].all(): forbidden[i, rng.integers(num_machines)] = 0
    return SchedulerState(machines=machines, apps=apps,
                          affinity=affinity, forbidden=forbidden)


# ============================================================================
# CLI — standalone run with configurable debate rounds
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="HPC Multi-Agent Debate Scheduler (4 agents, configurable debate)")
    p.add_argument("--model",          default="llama3")
    p.add_argument("--ollama-url",     default="http://localhost:11434")
    p.add_argument("--max-iter",       type=int,   default=1000)
    p.add_argument("--eps-rel",        type=float, default=0.0)
    p.add_argument("--eps-min",        type=float, default=0.25)
    p.add_argument("--debate-rounds",  type=int,   default=3,
                   help="Number of debate rounds (0=no debate, 3, 5, etc.)")
    p.add_argument("--num-choices",    type=int,   default=1,
                   help="Choices per advocate (1=single, 3=top-3, 0=full rank)")
    p.add_argument("--output-dir",     default="results_debate")
    p.add_argument("--verbose",        action="store_true")
    # CSV files
    p.add_argument("--apps",           default="Applications.csv")
    p.add_argument("--machines",       default="Machines.csv")
    p.add_argument("--affinity",       default="Affinity.csv")
    p.add_argument("--forbidden",      default="AntiAffinity.csv")
    # Subset / synthetic
    p.add_argument("--num-apps",       type=int, default=None)
    p.add_argument("--num-machines",   type=int, default=None)
    p.add_argument("--synthetic",      action="store_true")
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    if args.synthetic:
        num_apps     = args.num_apps     or 10
        num_machines = args.num_machines or 20
        state = generate_random_state(num_apps, num_machines, seed=args.seed)
        print(f"Synthetic mode: {num_apps} apps x {num_machines} machines (seed={args.seed})")
    else:
        apps_df     = pd.read_csv(args.apps)
        machines_df = pd.read_csv(args.machines)
        affinity_df = pd.read_csv(args.affinity,  index_col=0)
        forbidden_df= pd.read_csv(args.forbidden, index_col=0)

        num_apps     = args.num_apps     or len(apps_df)
        num_machines = args.num_machines or len(machines_df)
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
        print(f"CSV subset: {num_apps} apps x {num_machines} machines "
              f"({total_inst} total instances)")

    orch = HPCDebateOrchestrator(
        state=state, model=args.model, ollama_url=args.ollama_url,
        max_iterations=args.max_iter, eps_rel=args.eps_rel, eps_min=args.eps_min,
        debate_rounds=args.debate_rounds,
        num_choices=args.num_choices,
        verbose=args.verbose, output_dir=args.output_dir,
    )
    orch.run()
