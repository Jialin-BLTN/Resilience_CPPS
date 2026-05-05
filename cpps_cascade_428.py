"""
cpps_cascade.py  —  Cyber-Physical Power System Resilience Assessment
======================================================================
Implements the four-step framework from the paper:

  Step 1  Power Grid Modeling
          ├─ Physical layer : RTS-24 + sgen PV, AC power flow
          ├─ Cyber layer    : PMU placement (full / N-1 observable / random)
          └─ Comm. graph    : communication nodes (PDC/RTU) with link topology

  Step 2  Coordinated Attack Scenario Generation
          ├─ Physical faults  : N-k line outage, generator forced outage
          └─ Cyber attacks    : targeted PMU outage (DoS via comm. node),
                                comm. node disruption (cascade of PMU losses),
                                relay mis-operation (false-trip / block-trip),
                                false data injection (FDI on line flow sensors)

  Step 3  Steady-State Cascade Model  (per-stage loop, AC PF)
          ├─ Solve AC power flow at each stage
          ├─ Bidirectional coupling: physical overload <-> cyber detection
          ├─ Thermal overload check  (tau pickup + tau_hi instant)
          ├─ Detection-aware activation: PMU coverage gates relay response
          ├─ Island survivability: proportional load shedding on islands
          └─ Stage state record: F_s (tripped set), c_g(s) (stage severity)

  Step 4  Resilience Metrics
          ├─ Sub-indices  : S_load, S_struct, S_protect, S_obs
          ├─ Aggregation  : CFR, ACS, S95, CRI
          └─ Breakdown    : per attack-type, per PMU-config treatment group

Steady-state note
-----------------
Every power flow call is a full AC steady-state solution (runpp).
There is no transient / dynamic simulation.  Each cascade "stage k"
represents a new steady-state equilibrium reached after one round of
protection actions, not a physical time step.

Design rationale for relay thresholds (tau / tau_hi)
-----------------------------------------------------
RTS-24 is designed with generous thermal margins; base-case line
utilisation peaks at only 60-70%.  Two changes raise system stress to
a realistic operating level:

  (a) build_rts24(load_scale=1.25) multiplies all loads by 1.25,
      pushing peak-hour line utilisation to ~85-95%.
  (b) sample_profiles clips load_scale to [0.85, 1.30], preventing
      artificially light snapshots that would suppress cascade triggering.

With these adjustments, a typical N-1 outage redistributes flows to
105-130% of thermal rating, placing most scenarios inside the relay
window [tau=100, tau_hi=130]:

    tau    = 100%   — conservative pickup consistent with aged or
                     de-rated lines; triggers ~60-70% of N-1 scenarios.
    tau_hi = 130%  — emergency rating; instant trip for severe N-1/N-2.
    K_line = 2     — two consecutive stages above tau before timed trip,
                     modelling ANSI 51 definite-time overcurrent delay.

G0 weak PMU baseline
---------------------
G0 (PMUPlacement.NONE) now represents a grid with only 5 randomly placed
PMUs — the weakest PMU deployment in the comparison set.  This replaces
the previous "no PMU + conventional relay fallback" design and creates a
clean four-level PMU gradient:

    G0  :  5  PMUs  (NONE   — weak baseline)
    GA  : 10  PMUs  (RANDOM — random placement)
    GB  : ~20 PMUs  (N_MINUS_1 — greedy N-1 observable coverage)
    GC  : all buses (FULL   — complete coverage)

Because G0 now has PMUs, the conventional relay fallback branches in
detect_overloads_aware() and detect_undervoltage_aware() are no longer
needed and have been removed.  All four groups go through identical
PMU-aware relay logic, making the comparison fully symmetric.

Changes vs previous version
----------------------------
  1. PMUPlacement.NONE: returns 5 random PMU buses instead of empty set.
  2. Removed conventional relay fallback from detect_overloads_aware().
  3. Removed conventional relay fallback from detect_undervoltage_aware().
  4. Updated docstrings to reflect new G0 semantics.

Dependencies
------------
  pip install pandapower numpy pandas networkx scikit-learn openpyxl
"""

from __future__ import annotations

import copy
import random
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class AttackType(Enum):
    NONE      = "none"
    FDI       = "fdi"        # false data injection on line flow sensors
    DOS_PMU   = "dos_pmu"    # directly disable target PMU buses
    DOS_COMM  = "dos_comm"   # disrupt a comm node, cascading PMU loss
    RELAY_FT  = "relay_ft"   # relay false-trip (malicious trip command)


class PMUPlacement(Enum):
    FULL      = "full"       # PMU on every bus
    N_MINUS_1 = "n_minus_1"  # greedy N-1 observable coverage
    RANDOM    = "random"     # n_pmu_random randomly chosen buses
    NONE      = "none"       # 5 random PMUs — weak baseline G0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelayConfig:
    """Physical relay / protection settings.

    Threshold rationale for RTS-24 with load_scale=1.25
    ----------------------------------------------------
    After scaling loads by 1.25, peak-hour line utilisation reaches
    85-95%.  A typical N-1 outage then redistributes flows to 105-130%,
    placing most scenarios inside the [tau, tau_hi] relay window.

        tau    = 100%   conservative pickup (aged/de-rated line setting)
        tau_hi = 130%  emergency rating, instant trip
        K_line = 2     two consecutive pickup stages before timed trip
                       (models ANSI 51 definite-time overcurrent delay)

    slack_cap_factor = 0.10
        Cap the ext_grid at 10% of total installed generation to
        approximate spinning reserve.  Without a cap the slack bus
        absorbs all post-trip imbalances and masks genuine overloads.
    """
    tau:                      float = 105.0   # pickup threshold (% of rated loading)
    tau_hi:                   float = 120.0   # instant-trip threshold (% of rated)
    K_line:                   int   = 2       # consecutive stages before timed trip
    Vmin_pickup:              float = 0.90    # UVLS pickup voltage (pu)
    Vmin_trip:                float = 0.85    # UVLS trip voltage (pu)
    K_uvls:                   int   = 2       # consecutive stages before UVLS fires
    uvls_shed_frac:           float = 0.05    # fraction of bus load shed per UVLS stage
    slack_cap_factor:         float = 0.10    # ext_grid MW cap as fraction of total gen
    slack_overload_threshold: float = 0.85    # fraction of cap that flags slack stress
    max_stages:               int   = 30      # maximum cascade stages per scenario
    pf_max_tries:             int   = 3       # power flow retries before failure
    emergency_shed_frac:      float = 0.20    # load fraction shed on each PF retry


@dataclass
class SeverityWeights:
    """Weights for composite scenario severity Ss (must sum to 1.0).

    S_load    — fraction of total load lost (dominant term)
    S_struct  — fraction of lines tripped (structural damage)
    S_protect — UVLS + island-deficit shed normalised by total load
    S_obs     — observability loss fraction (cyber impact on PMU coverage)
    """
    beta_load:    float = 0.55
    beta_struct:  float = 0.25
    beta_protect: float = 0.15
    beta_obs:     float = 0.05


@dataclass
class PMUConfig:
    """PMU placement and cyber-attack size parameters.

    G0 baseline note
    ----------------
    When placement=NONE, exactly 5 PMU buses are placed at random
    (seeded by cfg.seed for reproducibility).  This represents the
    weakest PMU deployment and replaces the previous no-PMU + conventional
    relay fallback design.
    """
    placement:        PMUPlacement = PMUPlacement.N_MINUS_1
    n_pmu_random:     int          = 10    # buses chosen when placement=RANDOM
    # G0 fixed PMU count
    n_pmu_none:       int          = 5     # buses placed when placement=NONE
    # Communication graph
    n_comm_nodes:     int          = 6     # number of PDC/RTU aggregation nodes
    links_per_node:   int          = 4     # average PMU buses per comm node
    comm_seed:        int          = 99
    # FDI bias parameters
    fdi_bias_max:     float        = 0.20  # max fractional flow bias (+/-)
    fdi_line_frac:    float        = 0.50  # fraction of monitored lines corrupted
    # Attack scope (number of targets per event)
    n_dos_pmu_buses:  int          = 3
    n_dos_comm_nodes: int          = 1
    n_relay_lines:    int          = 2
    n_fdi_lines:      int          = 4
    seed:             int          = 42


@dataclass
class ScenarioConfig:
    """Monte-Carlo scenario library parameters."""
    N:             int   = 1000
    mode:          str   = "MC"      # "MC" | "ENUM_N1"
    p_n2:          float = 0.20      # probability of N-2 line outage
    p_gen_outage:  float = 0.15      # probability of generator forced outage
    seed:          int   = 7
    T:             int   = 24        # hourly profile length
    season:        str   = "equinox" # "summer" | "equinox" | "winter"
    load_sigma:    float = 0.04      # per-hour Gaussian noise on load scale
    pv_sigma:      float = 0.08      # per-hour Gaussian noise on PV CF
    spike_prob:    float = 0.06      # per-hour probability of PV irradiance spike
    spike_mag:     float = 0.35      # magnitude of PV spike/drop
    # Cyber event fractions
    cyber_fraction: float = 0.35
    p_fdi:          float = 0.25
    p_dos_pmu:      float = 0.25
    p_dos_comm:     float = 0.25
    p_relay_ft:     float = 0.25

    # Cyber attack scope
    n_dos_pmu_buses:  int   = 3
    n_dos_comm_nodes: int   = 1
    n_relay_lines:    int   = 2
    n_fdi_lines:      int   = 4
    fdi_bias_max:     float = 0.20

    # Optional real ERCOT profiles
    use_real_profiles:         bool          = False
    real_profile_lib:          object        = None
    real_load_sigma:           float         = 0.02
    real_solar_sigma:          float         = 0.03
    real_season_filter:        Optional[str] = None
    real_solar_cluster_filter: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SCENARIO & RESULT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CyberEvent:
    """Describes a single cyber attack attached to a scenario."""
    attack_type:  AttackType       = AttackType.NONE
    target_buses: List[int]        = field(default_factory=list)
    target_lines: List[int]        = field(default_factory=list)
    target_comm:  List[int]        = field(default_factory=list)
    fdi_biases:   Dict[int, float] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.attack_type != AttackType.NONE


@dataclass
class CyberState:
    """Mutable cyber state tracked inside run_cascade()."""
    downed_pmu_buses: Set[int]         = field(default_factory=set)
    blocked_lines:    Set[int]         = field(default_factory=set)
    false_tripped:    Set[int]         = field(default_factory=set)
    corrupted_lines:  Dict[int, float] = field(default_factory=dict)


@dataclass
class Scenario:
    """One disturbance scenario: outages + operating point + optional cyber event."""
    scenario_id:  int
    init_outages: List[Tuple[str, int]]
    t_index:      int
    load_scale:   np.ndarray             # shape (T,) normalised load multiplier
    pv_cf:        np.ndarray             # shape (T,) PV capacity factor
    source_date:  str        = "synthetic"
    cyber_event:  CyberEvent = field(default_factory=CyberEvent)


@dataclass
class StageRecord:
    """Snapshot of cascade state after stage k — maps to F_s in the paper."""
    stage:         int
    tripped_lines: List[int]
    p_served_mw:   float
    p_shed_mw:     float    # cumulative shed at this stage
    s_stage:       float    # normalised severity S(k) = p_shed / P0


@dataclass
class CascadeResult:
    """Full output record for one (scenario, treatment-group) simulation."""
    scenario_id:           int
    success_pf:            bool
    n_stages:              int
    tripped_lines:         List[int]
    uvls_shed_mw:          float
    island_shed_mw:        float
    slack_overload_stages: int
    severity:              float
    severity_components:   Dict[str, float]
    stage_records:         List[StageRecord]
    t_index:               int = 0
    source_date:           str = "synthetic"
    attack_type:           str = "none"
    n_downed_pmus:         int = 0
    n_blocked_lines:       int = 0
    n_false_tripped:       int = 0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMMUNICATION GRAPH  (Step 1 — Cyber layer)
# ══════════════════════════════════════════════════════════════════════════════

class CommGraph:
    """
    Simplified PDC/RTU communication topology.

    PMU buses are partitioned into clusters, each served by one comm node.
    The inter-node topology follows a Barabasi-Albert preferential attachment
    graph to approximate the hub-and-spoke structure of real SCADA networks.
    Disabling a comm node immediately silences all PMUs in its cluster.
    """

    def __init__(
        self,
        pmu_buses:      List[int],
        n_comm_nodes:   int,
        links_per_node: int,
        seed:           int,
    ) -> None:
        rng = np.random.default_rng(seed)
        n   = len(pmu_buses)
        # Clamp n_comm_nodes so we never create more nodes than PMU buses
        n_comm_nodes = min(n_comm_nodes, n)
        assignments = np.array_split(rng.permutation(pmu_buses), n_comm_nodes)
        self.node_to_pmus: Dict[int, List[int]] = {
            i: list(grp) for i, grp in enumerate(assignments)
        }
        self.pmu_to_node: Dict[int, int] = {
            bus: node
            for node, buses in self.node_to_pmus.items()
            for bus in buses
        }
        self.graph = nx.barabasi_albert_graph(
            n_comm_nodes, min(2, n_comm_nodes - 1), seed=int(seed)
        )
        self.n_comm_nodes = n_comm_nodes
        print(
            f"[CommGraph] {n_comm_nodes} comm nodes  "
            f"covering {n} PMU buses  "
            f"avg {n/n_comm_nodes:.1f} buses/node"
        )

    def pmus_downed_by_comm(self, comm_nodes: List[int]) -> Set[int]:
        """Return PMU buses silenced by disabling the given comm nodes."""
        downed: Set[int] = set()
        for cn in comm_nodes:
            downed.update(self.node_to_pmus.get(cn, []))
        return downed

    def random_comm_targets(self, n: int, rng: np.random.Generator) -> List[int]:
        """Sample n comm nodes uniformly at random without replacement."""
        nodes = list(range(self.n_comm_nodes))
        return [int(x) for x in rng.choice(nodes, size=min(n, len(nodes)), replace=False)]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PMU NETWORK  (Step 1 — Cyber layer)
# ══════════════════════════════════════════════════════════════════════════════

class PMUNetwork:
    """
    PMU placement and observability engine.

    Observability rule (N-1 standard):
        Bus b is observable if b is in pmu_buses OR every neighbour of b
        has a PMU.  This is the standard linear observability condition.

    G0 placement (PMUPlacement.NONE):
        Places exactly cfg.n_pmu_none (default 5) PMUs at randomly chosen
        buses, seeded by cfg.seed.  This is the weakest deployment tier.
        All four groups (G0/GA/GB/GC) use the same PMU-aware relay logic —
        there is no special fallback for G0.
    """

    def __init__(self, net: pp.pandapowerNet, cfg: PMUConfig) -> None:
        self.cfg       = cfg
        self.pmu_buses = self._place(net, cfg)
        self._topo     = self._build_topo(net)
        self.coverage  = self._compute_coverage(self.pmu_buses, net)
        self.bus_load  = self._bus_load_map(net)
        self.total_load_mw = max(sum(self.bus_load.values()), 1e-9)

        if len(self.pmu_buses) > 0:
            self.comm = CommGraph(
                list(self.pmu_buses),
                cfg.n_comm_nodes,
                cfg.links_per_node,
                cfg.comm_seed,
            )
        else:
            self.comm = None

        obs = sum(1 for v in self.coverage.values() if v)
        print(
            f"[PMU] {cfg.placement.value}  "
            f"PMUs={len(self.pmu_buses)}  "
            f"observable={obs}/{len(net.bus)}"
        )

    def _place(self, net: pp.pandapowerNet, cfg: PMUConfig) -> Set[int]:
        """Select PMU bus locations according to the chosen placement strategy.

        NONE  : place cfg.n_pmu_none (default 5) buses at random — G0 baseline.
        RANDOM: place cfg.n_pmu_random buses at random — GA.
        N_MINUS_1: greedy degree-descending to achieve N-1 observability — GB.
        FULL  : every bus — GC.
        """
        buses = net.bus.index.tolist()
        rng   = np.random.default_rng(cfg.seed)

        if cfg.placement == PMUPlacement.NONE:
            # G0: fixed 5 random PMUs (weakest baseline)
            n = min(cfg.n_pmu_none, len(buses))
            return set(int(b) for b in rng.choice(buses, n, replace=False))

        if cfg.placement == PMUPlacement.FULL:
            return set(buses)

        if cfg.placement == PMUPlacement.RANDOM:
            n = min(cfg.n_pmu_random, len(buses))
            return set(int(b) for b in rng.choice(buses, n, replace=False))

        # N-1 observable: greedy degree-descending placement
        G      = self._build_topo(net)
        placed: Set[int] = set()

        def obs(b: int) -> bool:
            return b in placed or all(nb in placed for nb in G.neighbors(b))

        for b in sorted(buses, key=lambda x: G.degree(x), reverse=True):
            if all(obs(x) for x in buses):
                break
            placed.add(b)
        for b in buses:
            if not obs(b):
                placed.add(b)
        return placed

    @staticmethod
    def _build_topo(net: pp.pandapowerNet) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(net.bus.index.tolist())
        for _, r in net.line.iterrows():
            G.add_edge(int(r["from_bus"]), int(r["to_bus"]))
        for _, r in net.trafo.iterrows():
            G.add_edge(int(r["hv_bus"]), int(r["lv_bus"]))
        return G

    def _compute_coverage(
        self, active_pmus: Set[int], net: pp.pandapowerNet
    ) -> Dict[int, bool]:
        G = self._topo
        return {
            int(b): (
                b in active_pmus
                or all(nb in active_pmus for nb in G.neighbors(b))
            )
            for b in net.bus.index
        }

    @staticmethod
    def _bus_load_map(net: pp.pandapowerNet) -> Dict[int, float]:
        m: Dict[int, float] = {}
        for _, r in net.load.iterrows():
            b = int(r["bus"])
            m[b] = m.get(b, 0.0) + float(r["p_mw"])
        return m

    def observable_buses(
        self, net: pp.pandapowerNet, downed: Set[int]
    ) -> Set[int]:
        active = self.pmu_buses - downed
        cov    = self._compute_coverage(active, net)
        return {b for b, v in cov.items() if v}

    def obs_loss_fraction(
        self, net: pp.pandapowerNet, downed: Set[int]
    ) -> float:
        """Fraction of total load on buses that lost observability."""
        if not downed:
            return 0.0
        obs  = self.observable_buses(net, downed)
        lost = sum(
            self.bus_load.get(b, 0.0)
            for b in net.bus.index if int(b) not in obs
        )
        return float(np.clip(lost / self.total_load_mw, 0.0, 1.0))

    def monitored_lines(
        self, net: pp.pandapowerNet, downed: Set[int]
    ) -> Set[int]:
        """
        Return lines whose BOTH endpoint buses are observable.
        Only monitored lines can trigger relay actions (PMU-aware coupling).
        """
        obs = self.observable_buses(net, downed)
        mon: Set[int] = set()
        for lid, row in net.line.iterrows():
            if int(row["from_bus"]) in obs and int(row["to_bus"]) in obs:
                mon.add(int(lid))
        return mon


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PHYSICAL NETWORK SETUP  (Step 1 — Physical layer)
# ══════════════════════════════════════════════════════════════════════════════

def build_rts24(load_scale: float = 1.25) -> pp.pandapowerNet:
    """
    Load the IEEE RTS-24 bus system and apply a global load scaling factor.

    RTS-24 was designed in 1979 with generous thermal margins; base-case
    line utilisation peaks at only 60-70%.  Scaling loads by 1.25 raises
    peak utilisation to ~85-95%, making tau=100% a physically realistic
    pickup threshold.
    """
    net = pn.case24_ieee_rts()
    if load_scale != 1.0:
        net.load["p_mw"]   *= load_scale
        net.load["q_mvar"] *= load_scale
    return net


def add_pv_as_sgen(
    net:         pp.pandapowerNet,
    n_pv:        int   = 5,
    pv_pen_frac: float = 0.15,
    seed:        int   = 7,
) -> List[int]:
    """
    Add PV generators as sgen (PQ nodes) to avoid slack-bus voltage conflict.

    Placement: load buses without existing synchronous generation are
    preferred, to avoid co-locating PV with voltage-controlling units.
    Each unit is initialised at p_mw=0 and dispatched via dispatch_pv().
    Reactive power is held at zero (unity PF, grid-following inverter).
    """
    rng        = np.random.default_rng(seed)
    gen_buses  = set(net.gen["bus"].tolist())
    load_buses = net.load["bus"].unique().tolist()
    candidates = [b for b in load_buses if b not in gen_buses] or load_buses
    chosen     = rng.choice(candidates, min(n_pv, len(candidates)), replace=False)
    total_cap  = float(
        net.gen["max_p_mw"].fillna(net.gen["p_mw"].clip(lower=0) + 1e-3).sum()
    )
    cap_each = pv_pen_frac * total_cap / len(chosen)
    ids = [
        int(pp.create_sgen(
            net, bus=int(b), p_mw=0.0, q_mvar=0.0,
            name=f"PV_bus{b}", type="PV",
            max_p_mw=cap_each, min_p_mw=0.0, in_service=True,
        ))
        for b in chosen
    ]
    print(f"[PV] {len(ids)} sgen units  rated {pv_pen_frac*total_cap:.1f} MW total")
    return ids


def dispatch_pv(
    net: pp.pandapowerNet, pv_ids: List[int], pv_cf: np.ndarray, t: int
) -> None:
    """Set PV p_mw = max_p_mw * cf[t].  Q forced to zero (unity PF)."""
    cf = float(np.clip(pv_cf[t], 0.0, 1.0))
    for sid in pv_ids:
        if sid in net.sgen.index and bool(net.sgen.at[sid, "in_service"]):
            net.sgen.at[sid, "p_mw"]   = float(net.sgen.at[sid, "max_p_mw"]) * cf
            net.sgen.at[sid, "q_mvar"] = 0.0


def cap_slack(net: pp.pandapowerNet, cfg: RelayConfig) -> float:
    """
    Cap ext_grid active power to approximate spinning reserve.

    Without a cap the slack bus absorbs all post-trip imbalances, masking
    genuine overloads and producing unrealistically low cascade severity.
    """
    total = float(
        net.gen["max_p_mw"].fillna(net.gen["p_mw"].clip(lower=0) + 1e-3).sum()
    )
    cap = cfg.slack_cap_factor * total
    if len(net.ext_grid) > 0:
        net.ext_grid["max_p_mw"] =  cap
        net.ext_grid["min_p_mw"] = -cap
        if net.ext_grid["max_q_mvar"].isna().any():
            net.ext_grid["max_q_mvar"] =  cap * 0.60
            net.ext_grid["min_q_mvar"] = -cap * 0.60
    return cap


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LOAD & PV PROFILES
# ══════════════════════════════════════════════════════════════════════════════

_LOAD_PARAMS: Dict[str, Tuple[float, float, float]] = {
    "industrial":  (0.10, 0.05, 0.92),
    "commercial":  (0.22, 0.18, 0.85),
    "residential": (0.12, 0.28, 0.75),
}

_SEASON_PV: Dict[str, Tuple] = {
    "summer":  (5.5,  20.0, 13.5, 1.00, 1.0, 2.2),
    "equinox": (6.5,  18.5, 13.0, 0.90, 1.2, 2.0),
    "winter":  (7.5,  16.5, 12.5, 0.70, 1.5, 2.5),
}


def dual_peak_load_shape(T: int, load_type: str = "residential") -> np.ndarray:
    """Normalised load shape with morning and evening peaks."""
    w_m, w_e, base = _LOAD_PARAMS.get(load_type, _LOAD_PARAMS["residential"])
    t = np.arange(T, dtype=float)
    s = (
        base
        + w_m  * np.exp(-0.5 * ((t - 9.0)  / 1.5) ** 2)
        + w_e  * np.exp(-0.5 * ((t - 19.0) / 2.0) ** 2)
        + 0.06 * np.exp(-0.5 * ((t - 13.0) / 1.0) ** 2)
    )
    return np.clip(s / s.max(), 0.50, 1.05)


def asymmetric_pv_cf(T: int, season: str = "equinox") -> np.ndarray:
    """Asymmetric PV capacity factor profile (power-law shape)."""
    sr, ss, peak, mx, re, fe = _SEASON_PV.get(season, _SEASON_PV["equinox"])
    t  = np.arange(T, dtype=float)
    cf = np.zeros(T)
    for i, ti in enumerate(t):
        if sr <= ti <= peak:
            cf[i] = mx * ((ti - sr) / (peak - sr)) ** re
        elif peak < ti <= ss:
            cf[i] = mx * ((ss - ti) / (ss - peak)) ** fe
    return cf


def classify_load_buses(net: pp.pandapowerNet) -> Dict[int, str]:
    """Classify each load bus as industrial / commercial / residential."""
    loads    = net.load[["bus", "p_mw"]].copy()
    q25, q75 = loads["p_mw"].quantile(0.25), loads["p_mw"].quantile(0.75)
    out: Dict[int, str] = {}
    for _, row in loads.iterrows():
        if row["p_mw"] >= q75:
            out[int(row["bus"])] = "industrial"
        elif row["p_mw"] <= q25:
            out[int(row["bus"])] = "residential"
        else:
            out[int(row["bus"])] = "commercial"
    return out


def build_load_shape_matrix(
    net: pp.pandapowerNet, T: int, bus_type_map: Dict[int, str]
) -> np.ndarray:
    """Build a (n_loads x T) matrix of normalised per-hour load multipliers."""
    cache = {
        lt: dual_peak_load_shape(T, lt)
        for lt in ("industrial", "commercial", "residential")
    }
    mat = np.ones((len(net.load), T))
    for i, (_, row) in enumerate(net.load.iterrows()):
        lt     = bus_type_map.get(int(row["bus"]), "commercial")
        mat[i] = cache[lt]
    return mat


def sample_profiles(
    cfg: ScenarioConfig,
    mat: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (load_scale, pv_cf) for one scenario."""
    if cfg.use_real_profiles and cfg.real_profile_lib is not None:
        lib        = cfg.real_profile_lib
        sol, ls, _ = lib.sample_day(
            rng, cfg.real_season_filter, cfg.real_solar_cluster_filter
        )
        ls  = np.clip(
            ls  + rng.normal(0, cfg.real_load_sigma, 24).astype(np.float32),
            0.65, 1.10,
        )
        sol = np.clip(
            sol + rng.normal(0, cfg.real_solar_sigma, 24).astype(np.float32),
            0.00, 1.00,
        )
        sol[sol < 0.01] = 0.0
        return ls, sol

    T       = cfg.T
    base_ls = mat.mean(axis=0)
    ls      = np.clip(base_ls + rng.normal(0, cfg.load_sigma, T), 0.85, 1.30)

    base_pv = asymmetric_pv_cf(T, cfg.season)
    pv      = base_pv.copy()
    dm      = base_pv > 0.02
    if dm.any():
        pv[dm] = np.clip(
            base_pv[dm] + rng.normal(0, cfg.pv_sigma, dm.sum()), 0.0, 1.10
        )
    for i in np.where(dm)[0]:
        if rng.random() < cfg.spike_prob:
            pv[i] = max(0.0, pv[i] - rng.uniform(0.15, cfg.spike_mag))
    return ls, pv


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SCENARIO GENERATION  (Step 2)
# ══════════════════════════════════════════════════════════════════════════════

def _make_physical_outages(
    net: pp.pandapowerNet,
    cfg: ScenarioConfig,
    rng: np.random.Generator,
) -> List[Tuple[str, int]]:
    """Draw one initiating physical disturbance."""
    line_ids = net.line.index.tolist()
    gen_ids  = net.gen.index.tolist()
    r        = rng.random()

    if r < cfg.p_gen_outage and gen_ids:
        return [("gen", int(rng.choice(gen_ids)))]
    elif r < cfg.p_gen_outage + cfg.p_n2 and len(line_ids) >= 2:
        l1, l2 = rng.choice(line_ids, size=2, replace=False)
        return [("line", int(l1)), ("line", int(l2))]
    else:
        return [("line", int(rng.choice(line_ids)))]


def _make_cyber_event(
    cfg:  ScenarioConfig,
    pmu:  PMUNetwork,
    rng:  np.random.Generator,
    net:  pp.pandapowerNet,
) -> CyberEvent:
    """Sample one coordinated cyber attack."""
    pmu_list = list(pmu.pmu_buses)
    line_ids = net.line.index.tolist()

    weights   = np.array([
        cfg.p_fdi, cfg.p_dos_pmu, cfg.p_dos_comm, cfg.p_relay_ft,
    ])
    weights  /= weights.sum()
    atype_map = [
        AttackType.FDI, AttackType.DOS_PMU, AttackType.DOS_COMM,
        AttackType.RELAY_FT,
    ]
    atype = atype_map[int(rng.choice(4, p=weights))]

    if atype == AttackType.FDI:
        mon_lines = list(pmu.monitored_lines(net, set()))
        n         = min(cfg.n_fdi_lines, len(mon_lines)) if mon_lines else 0
        tgt_lines = (
            [int(l) for l in rng.choice(mon_lines, n, replace=False)]
            if n > 0 else []
        )
        biases = {
            l: float(rng.uniform(-cfg.fdi_bias_max, cfg.fdi_bias_max))
            for l in tgt_lines
        }
        return CyberEvent(attack_type=atype, target_lines=tgt_lines, fdi_biases=biases)

    elif atype == AttackType.DOS_PMU:
        n   = min(cfg.n_dos_pmu_buses, len(pmu_list))
        tgt = (
            [int(b) for b in rng.choice(pmu_list, n, replace=False)]
            if n > 0 else []
        )
        return CyberEvent(attack_type=atype, target_buses=tgt)

    elif atype == AttackType.DOS_COMM:
        if pmu.comm is None:
            return CyberEvent()
        tgt_comm = pmu.comm.random_comm_targets(cfg.n_dos_comm_nodes, rng)
        downed   = list(pmu.comm.pmus_downed_by_comm(tgt_comm))
        return CyberEvent(attack_type=atype, target_buses=downed, target_comm=tgt_comm)

    elif atype == AttackType.RELAY_FT:
        n   = min(cfg.n_relay_lines, len(line_ids))
        tgt = [int(l) for l in rng.choice(line_ids, n, replace=False)]
        return CyberEvent(attack_type=atype, target_lines=tgt)

    return CyberEvent()


def generate_scenarios(
    net: pp.pandapowerNet,
    cfg: ScenarioConfig,
    pmu: PMUNetwork,
    mat: np.ndarray,
) -> List[Scenario]:
    """
    Build the shared scenario library for all treatment groups.

    The same set is reused across G0/GA/GB/GC for a fair paired comparison.
    t_index is drawn from high-stress hours (top 40% of load-shape mean).
    """
    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    base_avg     = mat.mean(axis=0)
    synth_stress = np.where(base_avg >= np.quantile(base_avg, 0.60))[0]
    if len(synth_stress) == 0:
        synth_stress = np.arange(cfg.T)

    def _pick_t(ls: np.ndarray, pv: np.ndarray) -> int:
        T    = len(ls)
        ramp = np.zeros(T)
        ramp[1:-1] = np.abs(pv[2:] - pv[:-2]) / 2.0
        score = ls * (1.0 + ramp)
        cands = np.where(score >= np.quantile(score, 0.60))[0]
        return int(rng.choice(cands if len(cands) > 0 else np.arange(T)))

    scenarios:  List[Scenario] = []
    outage_sets = (
        [[("line", int(i))] for i in net.line.index]
        if cfg.mode == "ENUM_N1" else None
    )

    n_total = cfg.N if cfg.mode == "MC" else len(outage_sets)
    for sid in range(n_total):
        outs   = (
            outage_sets[sid] if outage_sets is not None
            else _make_physical_outages(net, cfg, rng)
        )
        ls, pv = sample_profiles(cfg, mat, rng)
        t_idx  = (
            _pick_t(ls, pv) if cfg.use_real_profiles
            else int(rng.choice(synth_stress))
        )
        date   = (
            getattr(cfg.real_profile_lib, "_last_date", "unknown")
            if cfg.use_real_profiles else "synthetic"
        )
        evt = (
            _make_cyber_event(cfg, pmu, rng, net)
            if cfg.mode == "MC" and rng.random() < cfg.cyber_fraction
            else CyberEvent()
        )
        scenarios.append(Scenario(sid, outs, t_idx, ls, pv, date, evt))

    n_cyber     = sum(1 for s in scenarios if s.cyber_event.is_active())
    type_counts: Dict[str, int] = {}
    for s in scenarios:
        k = s.cyber_event.attack_type.value
        type_counts[k] = type_counts.get(k, 0) + 1
    print(f"[Scenarios] {len(scenarios)} total  cyber={n_cyber}  {type_counts}")
    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# 9.  OPERATING POINT APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def apply_load_scaling(
    net:    pp.pandapowerNet,
    base_p: pd.Series,
    base_q: pd.Series,
    mat:    np.ndarray,
    t:      int,
) -> None:
    """Scale each load to base_p[i] * shape_matrix[i, t], preserving PF."""
    for i, idx in enumerate(net.load.index):
        s = float(mat[i, t])
        net.load.at[idx, "p_mw"]   = float(base_p.iloc[i]) * s
        net.load.at[idx, "q_mvar"] = float(base_q.iloc[i]) * s


def apply_initial_outages(
    net:     pp.pandapowerNet,
    outages: List[Tuple[str, int]],
) -> None:
    """Set in_service=False for initiating event components."""
    tables = {"line": net.line, "gen": net.gen, "trafo": net.trafo}
    for typ, idx in outages:
        tbl = tables.get(typ)
        if tbl is not None and idx in tbl.index:
            tbl.at[idx, "in_service"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 10.  POWER FLOW + ISLAND HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def _conn_graph(net: pp.pandapowerNet) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(net.bus.index.tolist())
    for _, r in net.line[net.line["in_service"]].iterrows():
        G.add_edge(int(r["from_bus"]), int(r["to_bus"]))
    for _, r in net.trafo[net.trafo["in_service"]].iterrows():
        G.add_edge(int(r["hv_bus"]), int(r["lv_bus"]))
    return G


def island_load_shed(net: pp.pandapowerNet) -> float:
    """Proportional load shedding on islands without the slack bus."""
    if len(net.ext_grid) == 0:
        return 0.0
    slack_buses = set(net.ext_grid["bus"].tolist())
    shed = 0.0
    for comp in nx.connected_components(_conn_graph(net)):
        if comp & slack_buses:
            continue
        buses   = list(comp)
        p_gen   = (
            float(net.gen.loc[
                net.gen["bus"].isin(buses) & net.gen["in_service"], "p_mw"
            ].sum())
            + float(net.sgen.loc[
                net.sgen["bus"].isin(buses) & net.sgen["in_service"], "p_mw"
            ].sum())
        )
        lm      = net.load["bus"].isin(buses) & net.load["in_service"]
        p_load  = float(net.load.loc[lm, "p_mw"].sum())
        deficit = p_load - p_gen
        if deficit <= 1e-3:
            continue
        frac = min(1.0, deficit / p_load) if p_load > 1e-6 else 1.0
        for lid in net.load.index[lm]:
            p0 = float(net.load.at[lid, "p_mw"])
            q0 = float(net.load.at[lid, "q_mvar"])
            net.load.at[lid, "p_mw"]   = p0 * (1 - frac)
            net.load.at[lid, "q_mvar"] = q0 * (1 - frac)
            shed += p0 * frac
    return shed


def safe_runpp(
    net:       pp.pandapowerNet,
    cfg:       RelayConfig,
    slack_cap: float,
) -> bool:
    """AC power flow with graceful fallback."""
    for attempt in range(cfg.pf_max_tries):
        try:
            pp.runpp(
                net,
                init="auto",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=False,
            )
            return True
        except Exception:
            if attempt == 0:
                island_load_shed(net)
            else:
                net.load["p_mw"]   *= (1.0 - cfg.emergency_shed_frac)
                net.load["q_mvar"] *= (1.0 - cfg.emergency_shed_frac)
    return False


def check_slack_overload(net: pp.pandapowerNet, cap: float, thr: float) -> bool:
    """True if ext_grid injection exceeds thr * cap (slack stress indicator)."""
    if net.res_ext_grid is None or net.res_ext_grid.empty:
        return False
    return float(net.res_ext_grid["p_mw"].abs().max()) > thr * cap


# ══════════════════════════════════════════════════════════════════════════════
# 11.  CYBER ATTACK INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def initialise_cyber(
    net:         pp.pandapowerNet,
    event:       CyberEvent,
    pmu:         PMUNetwork,
    cyber_state: CyberState,
) -> float:
    """Apply the cyber event before the cascade loop starts."""
    if not event.is_active():
        return 0.0

    shed = 0.0

    if event.attack_type in (AttackType.DOS_PMU, AttackType.DOS_COMM):
        cyber_state.downed_pmu_buses.update(
            b for b in event.target_buses if b in pmu.pmu_buses
        )

    elif event.attack_type == AttackType.FDI:
        cyber_state.corrupted_lines.update(event.fdi_biases)

    elif event.attack_type == AttackType.RELAY_FT:
        for lid in event.target_lines:
            if lid in net.line.index and bool(net.line.at[lid, "in_service"]):
                net.line.at[lid, "in_service"] = False
                cyber_state.false_tripped.add(lid)
        shed = island_load_shed(net)

    return shed


# ══════════════════════════════════════════════════════════════════════════════
# 12.  DETECTION-AWARE RELAY LOGIC  (Step 3 — Bidirectional coupling)
# ══════════════════════════════════════════════════════════════════════════════

def detect_overloads_aware(
    net:         pp.pandapowerNet,
    relay_cfg:   RelayConfig,
    pmu:         PMUNetwork,
    cyber_state: CyberState,
) -> Tuple[Set[int], Set[int]]:
    """
    Identify overloaded lines respecting PMU observability.

    All four groups (G0/GA/GB/GC) now have PMUs, so relay actions are
    always gated by PMU coverage.  Only lines whose both endpoint buses
    are observable can trigger relay actions.  FDI corrupts apparent
    loading: observed_loading = true_loading * (1 + bias).

    Returns
    -------
    over    : lines with apparent loading > tau    (pickup)
    over_hi : lines with apparent loading > tau_hi (instant trip)
    """
    if net.res_line is None or net.res_line.empty:
        return set(), set()

    mon_lines = pmu.monitored_lines(net, cyber_state.downed_pmu_buses)

    over:    Set[int] = set()
    over_hi: Set[int] = set()

    for lid in net.res_line.index:
        if lid not in mon_lines:
            continue
        true_load = float(net.res_line.at[lid, "loading_percent"])
        bias      = cyber_state.corrupted_lines.get(int(lid), 0.0)
        app_load  = true_load * (1.0 + bias)
        if app_load > relay_cfg.tau:
            over.add(int(lid))
        if app_load > relay_cfg.tau_hi:
            over_hi.add(int(lid))

    return over, over_hi


def detect_undervoltage_aware(
    net:         pp.pandapowerNet,
    vmin:        float,
    pmu:         PMUNetwork,
    cyber_state: CyberState,
) -> Set[int]:
    """
    Return buses below vmin that are observable.

    All four groups now have PMUs; buses are only visible to the UVLS
    scheme if their PMU coverage is intact.
    """
    if net.res_bus is None or net.res_bus.empty:
        return set()

    obs = pmu.observable_buses(net, cyber_state.downed_pmu_buses)

    return {
        int(b) for b in net.res_bus.index
        if int(b) in obs and float(net.res_bus.at[b, "vm_pu"]) < vmin
    }


def uvls_action(net: pp.pandapowerNet, buses: List[int], frac: float) -> float:
    """Shed frac of load at each UVLS bus. Returns total MW shed."""
    shed = 0.0
    for b in buses:
        for lid in net.load.index[net.load["bus"] == b]:
            p0 = float(net.load.at[lid, "p_mw"])
            q0 = float(net.load.at[lid, "q_mvar"])
            net.load.at[lid, "p_mw"]   = p0 * (1 - frac)
            net.load.at[lid, "q_mvar"] = q0 * (1 - frac)
            shed += p0 * frac
    return shed


def trip_lines(net: pp.pandapowerNet, ids: List[int]) -> None:
    """Set in_service=False for each line in ids."""
    for lid in ids:
        if lid in net.line.index:
            net.line.at[lid, "in_service"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 13.  SEVERITY COMPUTATION  (Step 4)
# ══════════════════════════════════════════════════════════════════════════════

def severity_components(
    net0:         pp.pandapowerNet,
    net_end:      pp.pandapowerNet,
    tripped:      List[int],
    uvls_mw:      float,
    island_mw:    float,
    obs_frac:     float,
    base_load_mw: float,
) -> Dict[str, float]:
    """Four normalised severity sub-indices."""
    cur      = float(net_end.load["p_mw"].sum())
    S_load   = float(np.clip(1.0 - cur / max(1e-9, base_load_mw), 0.0, 1.0))
    S_struct = float(np.clip(
        len(set(tripped)) / max(1, len(net0.line.index)), 0.0, 1.0
    ))
    S_prot   = float(np.clip(
        (uvls_mw + island_mw) / max(1e-9, base_load_mw), 0.0, 1.0
    ))
    S_obs    = float(np.clip(obs_frac, 0.0, 1.0))
    return dict(S_load=S_load, S_struct=S_struct, S_protect=S_prot, S_obs=S_obs)


def composite_severity(comps: Dict[str, float], w: SeverityWeights) -> float:
    """Weighted sum of sub-indices -> scalar Ss in [0, 1]."""
    return float(np.clip(
        w.beta_load    * comps["S_load"]
        + w.beta_struct  * comps["S_struct"]
        + w.beta_protect * comps["S_protect"]
        + w.beta_obs     * comps["S_obs"],
        0.0, 1.0,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 14.  STEADY-STATE CASCADE RUNNER  (Step 3 — per-stage loop)
# ══════════════════════════════════════════════════════════════════════════════

def run_cascade(
    net_base:  pp.pandapowerNet,
    relay_cfg: RelayConfig,
    sev_w:     SeverityWeights,
    slack_cap: float,
    pmu:       PMUNetwork,
    event:     CyberEvent,
) -> CascadeResult:
    """
    Quasi-steady-state cascading failure simulation.

    Each stage k = one AC power-flow equilibrium after a round of
    protection actions.  No transient dynamics.
    """
    net  = copy.deepcopy(net_base)
    net0 = copy.deepcopy(net_base)
    P0   = float(net0.load["p_mw"].sum())

    cyber   = CyberState()
    isl     = initialise_cyber(net, event, pmu, cyber)
    tripped: List[int] = list(cyber.false_tripped)

    line_pickup:   Dict[int, int] = {}
    bus_uv_pickup: Dict[int, int] = {}
    uvls_shed  = 0.0
    slack_cnt  = 0
    stage_recs: List[StageRecord] = []

    # ── Initial power flow ────────────────────────────────────────────────────
    ok = safe_runpp(net, relay_cfg, slack_cap)
    if not ok:
        obs_f = pmu.obs_loss_fraction(net, cyber.downed_pmu_buses)
        comps = dict(S_load=1.0, S_struct=0.0, S_protect=1.0, S_obs=obs_f)
        return CascadeResult(
            scenario_id=0, success_pf=False, n_stages=0,
            tripped_lines=tripped, uvls_shed_mw=0.0, island_shed_mw=isl,
            slack_overload_stages=0,
            severity=composite_severity(comps, sev_w),
            severity_components=comps, stage_records=stage_recs,
            attack_type=event.attack_type.value,
            n_downed_pmus=len(cyber.downed_pmu_buses),
            n_blocked_lines=len(cyber.blocked_lines),
            n_false_tripped=len(cyber.false_tripped),
        )

    # ── Per-stage cascade loop ────────────────────────────────────────────────
    for k in range(relay_cfg.max_stages):
        if k == 0:
            max_load = float(net.res_line["loading_percent"].max())
            print(f"  k=0 max_loading={max_load:.1f}%  tau={relay_cfg.tau}")

        if check_slack_overload(net, slack_cap, relay_cfg.slack_overload_threshold):
            slack_cnt += 1

        over, over_hi = detect_overloads_aware(net, relay_cfg, pmu, cyber)
        uv_pick = detect_undervoltage_aware(net, relay_cfg.Vmin_pickup, pmu, cyber)
        uv_trip = detect_undervoltage_aware(net, relay_cfg.Vmin_trip,   pmu, cyber)

        # Update persistence counters
        for lid in list(line_pickup.keys()):
            line_pickup[lid] = line_pickup[lid] + 1 if lid in over else 0
        for lid in over:
            line_pickup.setdefault(lid, 1)

        for b in list(bus_uv_pickup.keys()):
            bus_uv_pickup[b] = bus_uv_pickup[b] + 1 if b in uv_pick else 0
        for b in uv_pick:
            bus_uv_pickup.setdefault(b, 1)

        # Build action set (blocked lines excluded)
        to_trip: Set[int] = over_hi - cyber.blocked_lines
        for lid, cnt in line_pickup.items():
            if cnt >= relay_cfg.K_line and lid not in cyber.blocked_lines:
                to_trip.add(lid)

        uvls_buses = [
            b for b, cnt in bus_uv_pickup.items()
            if cnt >= relay_cfg.K_uvls and b in uv_trip
        ]

        # Record F_s(k) snapshot before applying actions
        p_served = float(net.load["p_mw"].sum())
        p_shed_k = P0 - p_served
        stage_recs.append(StageRecord(
            stage=k,
            tripped_lines=list(tripped),
            p_served_mw=p_served,
            p_shed_mw=p_shed_k,
            s_stage=float(np.clip(p_shed_k / max(1e-9, P0), 0.0, 1.0)),
        ))

        # Terminate when cascade stabilises
        if not to_trip and not uvls_buses:
            obs_f = pmu.obs_loss_fraction(net, cyber.downed_pmu_buses)
            comps = severity_components(net0, net, tripped, uvls_shed, isl, obs_f, P0)
            return CascadeResult(
                scenario_id=0, success_pf=True, n_stages=k,
                tripped_lines=tripped, uvls_shed_mw=uvls_shed, island_shed_mw=isl,
                slack_overload_stages=slack_cnt,
                severity=composite_severity(comps, sev_w),
                severity_components=comps, stage_records=stage_recs,
                attack_type=event.attack_type.value,
                n_downed_pmus=len(cyber.downed_pmu_buses),
                n_blocked_lines=len(cyber.blocked_lines),
                n_false_tripped=len(cyber.false_tripped),
            )

        if uvls_buses:
            uvls_shed += uvls_action(net, uvls_buses, relay_cfg.uvls_shed_frac)

        if to_trip:
            live = [l for l in to_trip if bool(net.line.at[l, "in_service"])]
            trip_lines(net, live)
            tripped.extend(live)
            isl += island_load_shed(net)

        ok = safe_runpp(net, relay_cfg, slack_cap)
        if not ok:
            obs_f = pmu.obs_loss_fraction(net, cyber.downed_pmu_buses)
            comps = dict(
                S_load    = 1.0,
                S_struct  = float(np.clip(
                    len(set(tripped)) / max(1, len(net0.line.index)), 0.0, 1.0
                )),
                S_protect = float(np.clip(
                    (uvls_shed + isl) / max(1e-9, P0), 0.0, 1.0
                )),
                S_obs = float(np.clip(obs_f, 0.0, 1.0)),
            )
            return CascadeResult(
                scenario_id=0, success_pf=False, n_stages=relay_cfg.max_stages,
                tripped_lines=tripped, uvls_shed_mw=uvls_shed, island_shed_mw=isl,
                slack_overload_stages=slack_cnt,
                severity=composite_severity(comps, sev_w),
                severity_components=comps, stage_records=stage_recs,
                attack_type=event.attack_type.value,
                n_downed_pmus=len(cyber.downed_pmu_buses),
                n_blocked_lines=len(cyber.blocked_lines),
                n_false_tripped=len(cyber.false_tripped),
            )

    # max_stages reached without stabilisation
    obs_f = pmu.obs_loss_fraction(net, cyber.downed_pmu_buses)
    comps = severity_components(net0, net, tripped, uvls_shed, isl, obs_f, P0)
    return CascadeResult(
        scenario_id=0, success_pf=True, n_stages=relay_cfg.max_stages,
        tripped_lines=tripped, uvls_shed_mw=uvls_shed, island_shed_mw=isl,
        slack_overload_stages=slack_cnt,
        severity=composite_severity(comps, sev_w),
        severity_components=comps, stage_records=stage_recs,
        attack_type=event.attack_type.value,
        n_downed_pmus=len(cyber.downed_pmu_buses),
        n_blocked_lines=len(cyber.blocked_lines),
        n_false_tripped=len(cyber.false_tripped),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 15.  RESILIENCE METRIC AGGREGATION  (Step 4)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(
    results: List[CascadeResult],
    epsilon: float = 0.01,
    alpha:   float = 0.50,
) -> Dict[str, float]:
    """
    CFR  = fraction of scenarios with severity > epsilon
    ACS  = mean severity conditional on cascade (S > epsilon)
    S95  = 95th-percentile severity (tail risk)
    CRI  = 1 - (alpha*CFR + (1-alpha)*ACS),  higher = better resilience
    """
    S   = np.array([r.severity for r in results], dtype=float)
    CFR = float(np.mean(S > epsilon))
    ACS = float(np.mean(S[S > epsilon])) if np.any(S > epsilon) else 0.0
    S95 = float(np.quantile(S, 0.95))
    CRI = float(np.clip(1.0 - (alpha * CFR + (1.0 - alpha) * ACS), 0.0, 1.0))
    return dict(N=len(results), CFR=CFR, ACS=ACS, S95=S95, CRI=CRI)


def aggregate_table(
    results: List[CascadeResult],
    epsilon: float = 0.01,
    alpha:   float = 0.50,
) -> pd.DataFrame:
    """Per-attack-type resilience breakdown."""
    rows = []
    for atype in ["none", "fdi", "dos_pmu", "dos_comm", "relay_ft", "relay_bt"]:
        sub = [r for r in results if r.attack_type == atype]
        if not sub:
            continue
        m = aggregate(sub, epsilon, alpha)
        rows.append({"attack_type": atype, **m})
    return pd.DataFrame(rows)


def treatment_group_table(
    group_results: Dict[str, List[CascadeResult]],
    epsilon: float = 0.01,
    alpha:   float = 0.50,
) -> pd.DataFrame:
    """Per-treatment-group resilience summary."""
    rows = []
    for label, res in group_results.items():
        m = aggregate(res, epsilon, alpha)
        rows.append({"group": label, **m})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 16.  ERCOT PROFILE LIBRARY  (optional real-data driver)
# ══════════════════════════════════════════════════════════════════════════════

class ERCOTProfileLibrary:
    """Historical ERCOT solar CF and load normalisation profiles."""
    _SEASON_MONTHS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall":   [9, 10, 11],
    }

    def __init__(
        self,
        solar_cf_path:  str,
        load_norm_path: str,
        metadata_path:  str,
        archetypes_dir: str = None,
    ):
        self.solar = np.load(solar_cf_path).astype(np.float32)
        self.load  = np.load(load_norm_path).astype(np.float32)
        self.meta  = pd.read_csv(metadata_path, parse_dates=["date"])
        self.n     = len(self.solar)
        self._last_date = "unknown"
        print(f"[ERCOT] {self.n} days  solar {self.solar.shape}")

    def _candidates(self, sf: Optional[str], cf: Optional[int]) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if sf:
            mask &= self.meta["month"].isin(
                self._SEASON_MONTHS.get(sf.lower(), list(range(1, 13)))
            ).values
        if cf is not None and "solar_cluster" in self.meta.columns:
            mask &= self.meta["solar_cluster"].values == cf
        idx = np.where(mask)[0]
        return idx if len(idx) > 0 else np.arange(self.n)

    def sample_day(
        self,
        rng:                  np.random.Generator,
        season_filter:        Optional[str] = None,
        solar_cluster_filter: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        idx = int(rng.choice(self._candidates(season_filter, solar_cluster_filter)))
        dv  = self.meta.iloc[idx]["date"]
        self._last_date = str(dv.date() if hasattr(dv, "date") else dv)
        return self.solar[idx].copy(), self.load[idx].copy(), self._last_date


# ══════════════════════════════════════════════════════════════════════════════
# 17.  RESULTS EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def results_to_dataframe(results: List[CascadeResult]) -> pd.DataFrame:
    """Flatten CascadeResult list into a tidy per-scenario DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "scenario_id":           r.scenario_id,
            "source_date":           r.source_date,
            "t_index":               r.t_index,
            "attack_type":           r.attack_type,
            "success_pf":            r.success_pf,
            "n_stages":              r.n_stages,
            "n_tripped_lines":       len(set(r.tripped_lines)),
            "uvls_shed_mw":          r.uvls_shed_mw,
            "island_shed_mw":        r.island_shed_mw,
            "slack_overload_stages": r.slack_overload_stages,
            "n_downed_pmus":         r.n_downed_pmus,
            "n_blocked_lines":       r.n_blocked_lines,
            "n_false_tripped":       r.n_false_tripped,
            "severity":              r.severity,
            **r.severity_components,
        })
    return pd.DataFrame(rows)


def stage_records_to_dataframe(results: List[CascadeResult]) -> pd.DataFrame:
    """Flatten StageRecord objects -> F_s timeline table."""
    rows = []
    for r in results:
        for sr in r.stage_records:
            rows.append({
                "scenario_id": r.scenario_id,
                "attack_type": r.attack_type,
                "stage":       sr.stage,
                "p_served_mw": sr.p_served_mw,
                "p_shed_mw":   sr.p_shed_mw,
                "s_stage":     sr.s_stage,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()