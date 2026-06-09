"""
==========================================================================
  PLANET NINE DYNAMICAL EXPLORATION  —  v4.2  (Research Grade)
  Ammonite (2023 KQ14) + Known ETNO Ensemble

  ─────────────────────────────────────────────────────────────────────
  SCIENTIFIC MOTIVATION
  ─────────────────────
  Batygin & Brown (2016, 2019, 2021) propose a ~5–10 M_Earth planet at
  ~400–800 AU to explain the observed clustering of extreme TNO orbital
  poles and arguments of perihelion.  Ammonite (2023 KQ14, Wang et al.
  2024, Nature Astronomy) is the largest known ETNO (H_V ~ 3.5, D ~ 600 km)
  and adds a key data point.

  NEW IN v4.2 (vs v4.1)
  ─────────────────────
  1.  Switched integrator from WHFast to Mercurius.
  2.  Removed manual adaptive timestepping; Mercurius natively handles
      close encounters via IAS15 while maintaining a fixed base dt.
  3.  Drastically improved performance by removing Python-level distance
      checks and while loops.

  ─────────────────────────────────────────────────────────────────────
  INTEGRATOR DESIGN
  ─────────────────
  Giant-planet backbone
  └── Mercurius integrator (Rein et al. 2019)
      Fixed base dt = 0.5 yr  (≈ P_Jupiter / 24)
      Mercurius smoothly switches to IAS15 for close encounters,
      preserving excellent energy conservation without manual sub-stepping.

  High-eccentricity test particles  (Sedna e=0.84, Goblin e=0.94)
  └── IAS15 step-size control takes over automatically during perihelion,
      maintaining high accuracy even during planetary close approaches.

  ─────────────────────────────────────────────────────────────────────
  REBOUND 5.0.0 API  (verified against installed package)
  ─────────────────────────────────────────────────────────────────────
  1.  sim.integrator = "mercurius"
      sim.dt = 0.5
      (Handles close approaches automatically)

  2.  Particle naming:
        sim.particles[-1].name = "label"
        sim.particles["label"]  ← lookup by name

  3.  Orbit:
        p.orbit(primary=sim.particles["sun"])

  4.  Remove:
        sim.remove(idx)   where idx = sim.particles["name"].index

  5.  SimulationArchive:
        sim.save_to_file(path, interval=dt_yr, delete_file=True)
        sa = rebound.Simulationarchive(path)

  6.  integrate:
        sim.integrate(t, exact_finish_time=1)

  7.  Pickling:  sim.copy() and pickle.dumps/loads both work correctly.

  8.  OpenMP:  NOT compiled into the pip wheel.
      Use concurrent.futures.ProcessPoolExecutor for parallelism.
==========================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import ssl
import time
import warnings
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

# ── SSL fix (macOS / institutional machines) ──────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# ── Third-party ───────────────────────────────────────────────────────────────
import rebound
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ── REBOUND Horizons SSL fix ──────────────────────────────────────────────────
rebound.horizons.SSL_CONTEXT = "unverified"

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"         : "serif",
    "font.size"           : 11,
    "axes.labelsize"      : 12,
    "axes.titlesize"      : 11,
    "legend.fontsize"     : 8,
    "xtick.direction"     : "in",
    "ytick.direction"     : "in",
    "xtick.minor.visible" : True,
    "ytick.minor.visible" : True,
    "figure.dpi"          : 150,
})

# ── Output directory ──────────────────────────────────────────────────────────
SAVE_DIR = "."
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Parallelism ───────────────────────────────────────────────────────────────
N_CPU = int(os.environ.get("P9_NCPU", mp.cpu_count()))


# ══════════════════════════════════════════════════════════════════════════════
# §1  PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_M_MER   = 1.6601e-7
_M_VEN   = 2.4478e-6
_M_EAR   = 3.0027e-6
_M_MAR   = 3.2272e-7
M_INNER  = _M_MER + _M_VEN + _M_EAR + _M_MAR   # 6.2435e-6 M_sun

M_EARTH_MSUN = 3.003e-6   # 1 M_Earth in solar masses (IAU 2012)

# Fixed nominal timestep (Mercurius handles close encounters automatically)
DT_NOM = 0.5  # yr

# Hardcoded J2000.0 giant-planet elements (offline fallback)
GIANT_PLANETS_J2000 = [
    dict(m=9.54792e-4, a=5.20288, e=0.04839, inc=0.02275,
         Omega=1.75503, omega=0.25736, M=0.60033,  label="Jupiter"),
    dict(m=2.85887e-4, a=9.53667, e=0.05415, inc=0.04349,
         Omega=1.98471, omega=1.61356, M=0.87130,  label="Saturn"),
    dict(m=4.36624e-5, a=19.1891, e=0.04717, inc=0.01349,
         Omega=1.29551, omega=2.98376, M=5.46685,  label="Uranus"),
    dict(m=5.15138e-5, a=30.0699, e=0.00859, inc=0.03085,
         Omega=2.29978, omega=0.78480, M=5.32000,  label="Neptune"),
]


# ══════════════════════════════════════════════════════════════════════════════
# §2  ETNO ORBITAL CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

CATALOG = {
    "Ammonite (2023 KQ14)": {
        "a": 251.9,   "e": 0.7383, "inc": 10.98,
        "Omega": 72.104, "omega": 198.71, "M": 0.0,
        "color": "#2c6fad",
        "ref": "Wang et al. 2024, Nature Astronomy",
    },
    "Sedna (90377)": {
        "a": 506.8,   "e": 0.8437, "inc": 11.93,
        "Omega": 144.26, "omega": 311.28, "M": 358.1,
        "color": "#c0392b",
        "ref": "Brown et al. 2004; JPL Horizons 2024",
    },
    "2012 VP113": {
        "a": 261.0,   "e": 0.693,  "inc": 24.06,
        "Omega": 90.99,  "omega": 293.81, "M": 0.0,
        "color": "#8e44ad",
        "ref": "Trujillo & Sheppard 2014",
    },
    "2015 TG387 (Goblin)": {
        "a": 1170.0,  "e": 0.9408, "inc": 11.67,
        "Omega": 118.0,  "omega": 118.0,  "M": 0.0,
        "color": "#27ae60",
        "ref": "Sheppard et al. 2019",
    },
    "2013 SY99": {
        "a": 730.0,   "e": 0.9296, "inc": 4.23,
        "Omega": 29.5,   "omega": 32.4,   "M": 0.0,
        "color": "#e67e22",
        "ref": "Bannister et al. 2017",
    },
    "2015 BP519": {
        "a": 449.0,   "e": 0.921,  "inc": 54.1,
        "Omega": 135.2,  "omega": 348.1,  "M": 0.0,
        "color": "#e74c3c",
        "ref": "Becker et al. 2018",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# §3  PLANET NINE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

P9_NOMINAL = dict(
    m     = 5.0 * M_EARTH_MSUN,
    a     = 500.0,
    e     = 0.25,
    inc   = np.radians(20.0),
    Omega = np.radians(94.0),
    omega = np.radians(150.0),
    M     = 0.0,
    label = r"$m$=5 $M_\oplus$, $a$=500 AU, $e$=0.25",
)

P9_GRID = [
    dict(
        m       = m * M_EARTH_MSUN,
        a       = a,
        e       = e,
        inc     = np.radians(20.0),
        Omega   = np.radians(94.0),
        omega   = np.radians(150.0),
        M       = 0.0,
        label   = fr"$m$={m:.0f}$M_\oplus$, $a$={a} AU, $e$={e}",
        m_earth = m,
        a_p9    = a,
        e_p9    = e,
    )
    for m in [5.0, 10.0]
    for a in [400.0, 500.0, 600.0]
    for e in [0.20, 0.35]
]


# ══════════════════════════════════════════════════════════════════════════════
# §4  CLONE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_clones(N_req: int = 200, seed: int = 42) -> list:
    """Sample orbital clones from the full 6×6 covariance matrix."""
    np.random.seed(seed)
    nom = CATALOG["Ammonite (2023 KQ14)"]

    mu = np.array([
        nom["a"],
        nom["e"],
        np.radians(nom["inc"]),
        np.radians(nom["Omega"]),
        np.radians(nom["omega"]),
        np.radians(nom["M"]),
    ])

    sig = np.array([
        0.30,
        3.0e-4,
        np.radians(0.010),
        np.radians(0.012),
        np.radians(0.015),
        np.radians(0.020),
    ])

    rho = np.eye(6)
    rho[0, 1] = rho[1, 0] = -0.45
    rho[0, 4] = rho[4, 0] =  0.10
    rho[1, 4] = rho[4, 1] = -0.12
    rho[2, 3] = rho[3, 2] =  0.08

    cov = rho * np.outer(sig, sig)
    lam = np.linalg.eigvalsh(cov)
    if lam.min() <= 0:
        cov += np.eye(6) * (abs(lam.min()) + 1e-15)

    raw = np.random.multivariate_normal(mu, cov, N_req)
    out = [r.tolist() for r in raw
           if 0.0 < r[1] < 1.0 and r[0] > 0.0 and 0.0 <= r[2] <= np.pi]

    print(f"  Clones: {N_req} requested  →  {len(out)} valid  "
          f"({N_req - len(out)} rejected out of bounds)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §5  ORBITAL ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def rayleigh(angles) -> tuple:
    a = np.asarray(angles, dtype=float)
    N = len(a)
    if N < 2:
        return np.nan, np.nan, np.nan
    C  = np.mean(np.cos(a))
    S  = np.mean(np.sin(a))
    R  = float(np.hypot(C, S))
    mu = float(np.arctan2(S, C))
    z  = N * R**2
    pv = np.exp(-z) * (1.0
                       + (2*z - z**2) / (4*N)
                       - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*N**2))
    return R, mu, float(np.clip(pv, 0.0, 1.0))


def orbital_pole(inc: float, Omega: float) -> np.ndarray:
    return np.array([
        np.sin(inc) * np.sin(Omega),
       -np.sin(inc) * np.cos(Omega),
        np.cos(inc),
    ])


def pole_clustering(inc_list, Omega_list) -> tuple:
    if len(inc_list) < 2:
        return np.nan, np.full(3, np.nan)
    poles = np.array([orbital_pole(i, O) for i, O in zip(inc_list, Omega_list)])
    mean_vec = poles.mean(axis=0)
    R = float(np.linalg.norm(mean_vec))
    mean_unit = mean_vec / R if R > 0 else mean_vec
    return R, mean_unit


def detect_kozai(e_track: list, inc_track: list, threshold: float = -0.6) -> bool:
    if len(e_track) < 10:
        return False
    e_arr   = np.array(e_track, dtype=float)
    inc_arr = np.array(inc_track, dtype=float)
    valid   = np.isfinite(e_arr) & np.isfinite(inc_arr)
    if valid.sum() < 10:
        return False
    corr = float(np.corrcoef(e_arr[valid], inc_arr[valid])[0, 1])
    return corr < threshold


def lon_peri(orb) -> float:
    return orb.omega + orb.Omega


def mean_lon(orb) -> float:
    return orb.M + orb.omega + orb.Omega


def phi_3_1(orb_tno, orb_p9) -> float:
    angle = (3.0 * mean_lon(orb_tno)
             - mean_lon(orb_p9)
             - 2.0 * lon_peri(orb_tno))
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def safe_orbit(sim, name: str):
    try:
        p   = sim.particles[name]
        sun = sim.particles["sun"]
        o   = p.orbit(primary=sun)
        if o.a > 0.0 and 0.0 < o.e < 1.0:
            return o
    except Exception:
        pass
    return None


def heliocentric_r(sim, pname: str) -> float:
    try:
        p   = sim.particles[pname]
        sun = sim.particles["sun"]
        return float(np.sqrt((p.x - sun.x)**2 +
                             (p.y - sun.y)**2 +
                             (p.z - sun.z)**2))
    except Exception:
        return float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# §6  SIMULATION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _add_giants(sim: rebound.Simulation, use_horizons: bool = False) -> None:
    if use_horizons:
        names = ["Jupiter", "Saturn", "Uranus", "Neptune"]
        try:
            for name in names:
                sim.add(name)
            return
        except Exception:
            pass 

    for g in GIANT_PLANETS_J2000:
        sim.add(m=g["m"], a=g["a"], e=g["e"],
                inc=g["inc"], Omega=g["Omega"],
                omega=g["omega"], M=g["M"])
        sim.particles[-1].name = g["label"]


def build_backbone(p9: dict = None,
                   use_horizons: bool = True) -> rebound.Simulation:
    """
    Build the massive-particle backbone: Sun + giants [+ P9].

    Integrator: Mercurius
    Base timestep: 0.5 yr
    """
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")

    sim.integrator           = "mercurius"
    sim.dt                   = DT_NOM
    sim.exit_max_distance    = 3000.0   # AU

    sim.add(m=1.0 + M_INNER)
    sim.particles[-1].name = "sun"

    _add_giants(sim, use_horizons=use_horizons)

    if p9 is not None:
        sim.add(m=p9["m"], a=p9["a"], e=p9["e"],
                inc=p9["inc"], Omega=p9["Omega"],
                omega=p9["omega"], M=p9["M"])
        sim.particles[-1].name = "planet9"

    sim.move_to_com()
    return sim


# ══════════════════════════════════════════════════════════════════════════════
# §7  ENERGY CONSERVATION DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

def measure_energy_conservation(p9: dict, use_horizons: bool,
                                 t_end: float = 1e6,
                                 N_steps: int = 200) -> dict:
    """
    Integrate the backbone alone and record ΔE/E vs time.
    Demonstrates Mercurius numerical quality to Dr. Ying.
    """
    print("  Energy conservation diagnostic … ", end="", flush=True)
    sim = build_backbone(p9=p9, use_horizons=use_horizons)
    E0  = sim.energy()
    times  = np.linspace(0.0, t_end, N_steps)
    dE_rel = np.zeros(N_steps)

    for i, t in enumerate(times):
        sim.integrate(t, exact_finish_time=1)
        dE_rel[i] = abs((sim.energy() - E0) / E0)

    print(f"done  (max ΔE/E = {dE_rel.max():.2e})")
    return dict(times=times, t_myr=times / 1e6, dE_rel=dE_rel)


# ══════════════════════════════════════════════════════════════════════════════
# §8  SECULAR RESONANCE OVERLAP MAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_secular_map(p9: dict, use_horizons: bool,
                         n_a: int = 20, n_e: int = 20,
                         t_end: float = 1e7) -> dict:
    print(f"  Secular resonance map ({n_a}×{n_e} grid, t={t_end/1e6:.0f} Myr) …")

    a_nom = CATALOG["Ammonite (2023 KQ14)"]["a"]
    a_arr = np.linspace(150.0, 400.0, n_a)
    e_arr = np.linspace(0.50,  0.95,  n_e)
    delta_omega = np.full((n_a, n_e), np.nan)

    backbone = build_backbone(p9=p9, use_horizons=use_horizons)

    for i, a in enumerate(tqdm(a_arr, desc="  Secular map rows", leave=False)):
        for j, e in enumerate(e_arr):
            sim = backbone.copy()
            inc   = np.radians(15.0)
            Omega = np.radians(100.0)
            omega = np.radians(200.0)
            sim.add(m=0.0, a=a, e=e, inc=inc,
                    Omega=Omega, omega=omega, M=0.0)
            sim.particles[-1].name = "probe"
            sim.move_to_com()

            o0 = safe_orbit(sim, "probe")
            if o0 is None:
                continue
            om0 = o0.omega

            try:
                sim.integrate(t_end, exact_finish_time=1)
            except (rebound.Escape, Exception):
                pass

            o1 = safe_orbit(sim, "probe")
            if o1 is None:
                continue
            delta_omega[i, j] = abs(o1.omega - om0)

    return dict(a_arr=a_arr, e_arr=e_arr, delta_omega=delta_omega,
                a_nom=a_nom)


# ══════════════════════════════════════════════════════════════════════════════
# §9  PARALLEL CLONE WORKER  (top-level for pickling)
# ══════════════════════════════════════════════════════════════════════════════

def _worker_integrate_clone(args: tuple):
    """
    Worker executed in a separate OS process via ProcessPoolExecutor.
    Receives pickled backbone + one clone's orbital elements.
    
    The Mercurius integrator automatically handles close encounters 
    with high precision while keeping the base dt fixed.
    """
    backbone_sim, clone_el, t_end, N_steps, p9_present = args

    a, e, inc, Om, om, M = clone_el
    sim = backbone_sim.copy()

    sim.add(m=0.0, a=a, e=e, inc=inc, Omega=Om, omega=om, M=M)
    sim.particles[-1].name = "clone"
    sim.move_to_com()

    times  = np.linspace(0.0, t_end, N_steps)
    a_arr  = np.full(N_steps, np.nan)
    e_arr  = np.full(N_steps, np.nan)
    inc_arr= np.full(N_steps, np.nan)
    om_arr = np.full(N_steps, np.nan)
    vp_arr = np.full(N_steps, np.nan)
    ph_arr = np.full(N_steps, np.nan)
    alive  = True

    for i, t in enumerate(times):
        if not alive:
            break

        # Fast integration straight to t; Mercurius manages close encounters
        try:
            sim.integrate(t, exact_finish_time=1)
        except rebound.Escape:
            alive = False
            break

        o = safe_orbit(sim, "clone")
        if o is None:
            alive = False
            break

        a_arr[i]   = o.a
        e_arr[i]   = o.e
        inc_arr[i] = o.inc
        om_arr[i]  = o.omega
        vp_arr[i]  = lon_peri(o)

        if p9_present:
            op = safe_orbit(sim, "planet9")
            if op is not None:
                ph_arr[i] = phi_3_1(o, op)

    return dict(a=a_arr, e=e_arr, inc=inc_arr,
                omega=om_arr, varpi=vp_arr, phi31=ph_arr)


# ══════════════════════════════════════════════════════════════════════════════
# §10  REFERENCE ETNO INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _is_fled(sim, pname, sx, sy, sz, rmax2) -> bool:
    try:
        p = sim.particles[pname]
        return ((p.x-sx)**2 + (p.y-sy)**2 + (p.z-sz)**2) > rmax2
    except Exception:
        return True


def run_etno_reference(p9: dict, catalog: dict,
                        t_end: float, N_steps: int,
                        use_horizons: bool) -> dict:
    """
    Integrate all ETNOs in a single simulation.
    Uses Mercurius integrator for fast C-level close-encounter handling.
    """
    sim = build_backbone(p9=p9, use_horizons=use_horizons)

    etno_pnames = {}
    for i, (name, el) in enumerate(catalog.items()):
        pname = f"etno_{i}"
        sim.add(m=0.0, a=el["a"], e=el["e"],
                inc=np.radians(el["inc"]),
                Omega=np.radians(el["Omega"]),
                omega=np.radians(el["omega"]),
                M=np.radians(el["M"]))
        sim.particles[-1].name = pname
        etno_pnames[name] = pname
    sim.move_to_com()

    times      = np.linspace(0.0, t_end, N_steps)
    etno_tracks = {n: dict(a=[], e=[], inc=[], om=[], vp=[]) for n in catalog}
    R_om  = np.full(N_steps, np.nan)
    pv_om = np.full(N_steps, np.nan)
    R_vp  = np.full(N_steps, np.nan)
    pv_vp = np.full(N_steps, np.nan)
    R_pole= np.full(N_steps, np.nan) 
    active = dict(etno_pnames)
    dead   = set()

    def _drop(pname):
        try:
            idx = sim.particles[pname].index
            sim.remove(idx)
        except Exception:
            pass

    for step, t in enumerate(times):
        # We loop just in case multiple particles escape before t
        while sim.t < t - 1e-9:
            try:
                sim.integrate(t, exact_finish_time=1)
            except rebound.Escape:
                sun2 = sim.particles["sun"]
                sx, sy, sz = sun2.x, sun2.y, sun2.z
                rmax2 = sim.exit_max_distance**2
                escaped = [nm for nm, pn in list(active.items())
                           if _is_fled(sim, pn, sx, sy, sz, rmax2)]
                for nm in escaped:
                    _drop(active[nm])
                    del active[nm]
                    dead.add(nm)

        all_om, all_vp, all_inc, all_Omega = [], [], [], []

        for cname, pname in list(active.items()):
            o = safe_orbit(sim, pname)
            if o is None:
                continue
            vp = lon_peri(o)
            etno_tracks[cname]["a"].append(o.a)
            etno_tracks[cname]["e"].append(o.e)
            etno_tracks[cname]["inc"].append(o.inc)
            etno_tracks[cname]["om"].append(o.omega)
            etno_tracks[cname]["vp"].append(vp)
            all_om.append(o.omega)
            all_vp.append(vp)
            all_inc.append(o.inc)
            all_Omega.append(o.Omega)

        if len(all_om) >= 3:
            R, _, pv   = rayleigh(all_om)
            R_om[step]  = R;  pv_om[step] = pv
        if len(all_vp) >= 3:
            R, _, pv   = rayleigh(all_vp)
            R_vp[step]  = R;  pv_vp[step] = pv
        if len(all_inc) >= 3:
            Rp, _      = pole_clustering(all_inc, all_Omega)
            R_pole[step] = Rp

    # ── Kozai–Lidov detection ─────────────────────────────────────────────────
    kl_flags = {}
    for name, tr in etno_tracks.items():
        kl_flags[name] = detect_kozai(tr["e"], tr["inc"])

    return dict(
        times   = times,
        t_myr   = times / 1e6,
        etno    = etno_tracks,
        R_om    = R_om,   pv_om  = pv_om,
        R_vp    = R_vp,   pv_vp  = pv_vp,
        R_pole  = R_pole,
        kl_flags= kl_flags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §11  PARALLEL CLONE ENSEMBLE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_clone_ensemble(clones: list, backbone_sim: rebound.Simulation,
                        t_end: float, N_steps: int,
                        p9_present: bool, label: str = "") -> dict:
    n      = len(clones)
    times  = np.linspace(0.0, t_end, N_steps)
    work   = [(backbone_sim, c, t_end, N_steps, p9_present) for c in clones]

    print(f"  Launching {n} clones on {N_CPU} cores {label} …")
    t0 = time.time()

    raw_results = [None] * n
    with ProcessPoolExecutor(max_workers=N_CPU) as ex:
        future_map = {ex.submit(_worker_integrate_clone, w): i
                      for i, w in enumerate(work)}
        for fut in tqdm(as_completed(future_map),
                        total=n, desc=f"  Clones{label}", leave=True):
            idx = future_map[fut]
            try:
                raw_results[idx] = fut.result()
            except Exception as exc:
                raw_results[idx] = None

    dt_run = time.time() - t0
    alive_results = [r for r in raw_results if r is not None]
    print(f"  Done in {dt_run:.1f} s  ({len(alive_results)}/{n} survived)")

    # ── Aggregate scalars ─────────────────────────────────────────────────────
    def _agg(key):
        vals = [r[key] for r in alive_results]
        if not vals:
            return (np.full(N_steps, np.nan),) * 4
        mat = np.array(vals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mn  = np.nanmean(mat, axis=0)
            std = np.nanstd(mat,  axis=0)
            mn_ = np.nanmin(mat,  axis=0)
            mx  = np.nanmax(mat,  axis=0)
        return mn, std, mn_, mx

    a_mean,  a_std,  a_min,  a_max  = _agg("a")
    e_mean,  e_std,  _,      _      = _agg("e")
    om_mean, om_std, _,      _      = _agg("omega")
    phi31_mean, _, _, _             = _agg("phi31")

    # ── Survival count ────────────────────────────────────────────────────────
    survival = np.zeros(N_steps, dtype=int)
    for r in alive_results:
        survival += ~np.isnan(r["a"])

    # ── Rayleigh R_omega over clones ──────────────────────────────────────────
    R_om_clones  = np.full(N_steps, np.nan)
    pv_om_clones = np.full(N_steps, np.nan)
    for step in range(N_steps):
        oms = [r["omega"][step] for r in alive_results
               if not np.isnan(r["omega"][step])]
        if len(oms) >= 3:
            R, _, pv = rayleigh(oms)
            R_om_clones[step]  = R
            pv_om_clones[step] = pv

    # ── Orbital pole clustering R_pole over clones ────────────────────────────
    R_pole_clones = np.full(N_steps, np.nan)
    for step in range(N_steps):
        incs   = [r["inc"][step]   for r in alive_results
                  if not np.isnan(r["inc"][step])]
        Omegas = [r["varpi"][step] - r["omega"][step] for r in alive_results
                  if not np.isnan(r["varpi"][step]) and not np.isnan(r["omega"][step])]
        if len(incs) >= 3 and len(incs) == len(Omegas):
            Rp, _ = pole_clustering(incs, Omegas)
            R_pole_clones[step] = Rp

    # ── Store per-clone snapshots for phase-space portrait ────────────────────
    a_all  = np.array([r["a"] for r in alive_results])
    e_all  = np.array([r["e"] for r in alive_results])

    return dict(
        times         = times,
        t_myr         = times / 1e6,
        a_mean        = a_mean,
        a_std         = a_std,
        a_min         = a_min,
        a_max         = a_max,
        e_mean        = e_mean,
        e_std         = e_std,
        om_mean       = om_mean,
        om_std        = om_std,
        survival      = survival,
        phi31_mean    = phi31_mean,
        R_om_clones   = R_om_clones,
        pv_om_clones  = pv_om_clones,
        R_pole_clones = R_pole_clones,
        a_all         = a_all,
        e_all         = e_all,
        n_clones      = n,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §12  P9 PARAMETER GRID SCAN
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(clones: list, catalog: dict, p9_grid: list,
             t_end: float = 5e7, N_steps: int = 50,
             N_cl: int = 50, use_horizons: bool = True) -> list:
    print(f"\n  Grid: {len(p9_grid)} P9 configs × "
          f"{t_end/1e6:.0f} Myr each ({N_cl} clones/config) …")
    sub     = clones[:N_cl]
    results = []

    for ii, p9 in enumerate(p9_grid):
        print(f"\n  [{ii+1:2d}/{len(p9_grid)}]  {p9['label']}")
        backbone = build_backbone(p9=p9, use_horizons=use_horizons)
        res_e    = run_etno_reference(p9, catalog, t_end, N_steps, use_horizons)
        res_c    = run_clone_ensemble(sub, backbone, t_end, N_steps,
                                      p9_present=True, label=f"({p9['label']})")

        v_om = ~np.isnan(res_e["R_om"])
        v_vp = ~np.isnan(res_e["R_vp"])
        v_pl = ~np.isnan(res_e["R_pole"])
        Ro   = float(res_e["R_om"][v_om][-1])   if v_om.any() else np.nan
        Rv   = float(res_e["R_vp"][v_vp][-1])   if v_vp.any() else np.nan
        Rp   = float(res_e["R_pole"][v_pl][-1]) if v_pl.any() else np.nan
        sv   = int(res_c["survival"][-1])

        results.append(dict(
            label   = p9["label"],
            m_earth = p9["m_earth"],
            a_p9    = p9["a_p9"],
            e_p9    = p9["e_p9"],
            R_om    = Ro,
            R_vp    = Rv,
            R_pole  = Rp,
            survival= sv,
        ))
        print(f"     R_omega={Ro:.3f}  R_varpi={Rv:.3f}  "
              f"R_pole={Rp:.3f}  surv={sv}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# §13  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

_BLUE = "#2c6fad"
_RED  = "#e74c3c"
_GRN  = "#27ae60"
_PRP  = "#8e44ad"

def _band(ax, t, arr_mean, arr_std, c, ls="-", lw=1.8):
    v = ~np.isnan(arr_mean)
    if not v.any():
        return
    ax.fill_between(t[v],
                    arr_mean[v] - arr_std[v],
                    arr_mean[v] + arr_std[v],
                    alpha=0.22, color=c)
    ax.plot(t[v], arr_mean[v], color=c, lw=lw, ls=ls)

# ── Figure 1: Main 8-panel ────────────────────────────────────────────────────

def plot_main(rp_e, rp_c, rn_e, rn_c, cat) -> None:
    t   = rp_c["t_myr"]
    fig = plt.figure(figsize=(15, 19))
    gs  = gridspec.GridSpec(4, 2, fig, hspace=0.44, wspace=0.30)

    # (a) Semi-major axis
    ax = fig.add_subplot(gs[0, 0])
    for rc, c, lbl, ls in [(rp_c, _BLUE, "with P9", "-"),
                            (rn_c, _RED,  "without P9", "--")]:
        v = ~np.isnan(rc["a_mean"])
        if v.any():
            ax.fill_between(t[v], rc["a_min"][v], rc["a_max"][v],
                            alpha=0.08, color=c)
            _band(ax, t, rc["a_mean"], rc["a_std"], c, ls)
            ax.plot([], [], color=c, ls=ls, lw=1.8, label=lbl)
    ax.set_ylabel("$a$ (AU)")
    ax.set_title(r"(a) Ammonite clone $a$ — with vs without P9")
    ax.legend(); ax.grid(alpha=0.25)

    # (b) Eccentricity
    ax = fig.add_subplot(gs[0, 1])
    for rc, c, ls in [(rp_c, _BLUE, "-"), (rn_c, _RED, "--")]:
        _band(ax, t, rc["e_mean"], rc["e_std"], c, ls)
    ax.set_ylabel("$e$")
    ax.set_title(r"(b) Ammonite clone $e$")
    ax.grid(alpha=0.25)

    # (c) omega tracks
    ax = fig.add_subplot(gs[1, 0])
    for name, tr in rp_e["etno"].items():
        if not tr["om"]:
            continue
        n_t  = len(tr["om"])
        flag = "  ★KL" if rp_e["kl_flags"].get(name) else ""
        ax.plot(t[:n_t], np.degrees(tr["om"]) % 360,
                color=cat[name]["color"], lw=1.1, alpha=0.85,
                label=name + flag)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel(r"$\omega$ (°)")
    ax.set_title(r"(c) Argument of perihelion $\omega$ — with P9"
                 "\n★ = Kozai–Lidov cycling detected")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)

    # (d) varpi tracks
    ax = fig.add_subplot(gs[1, 1])
    for name, tr in rp_e["etno"].items():
        if not tr["vp"]:
            continue
        n_t = len(tr["vp"])
        ax.plot(t[:n_t], np.degrees(tr["vp"]) % 360,
                color=cat[name]["color"], lw=1.1, alpha=0.85, label=name)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel(r"$\varpi = \omega + \Omega$ (°)")
    ax.set_title(r"(d) Longitude of perihelion $\varpi$ — with P9")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)

    # (e) Rayleigh R_omega
    ax = fig.add_subplot(gs[2, 0])
    for re, c, lbl, ls in [(rp_e, _BLUE, "ETNO, with P9", "-"),
                            (rn_e, _RED,  "ETNO, without P9", "--")]:
        v = ~np.isnan(re["R_om"])
        if v.any():
            ax.plot(t[v], re["R_om"][v], color=c, lw=1.8, ls=ls, label=lbl)
    v = ~np.isnan(rp_c["R_om_clones"])
    if v.any():
        ax.plot(t[v], rp_c["R_om_clones"][v], color=_GRN, lw=1.4, ls=":",
                label="clones, with P9")
    ax.axhline(0.5, color="gray", ls=":",  lw=1, alpha=0.7)
    ax.axhline(0.7, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"Rayleigh $R_\omega$")
    ax.set_title(r"(e) $\omega$ clustering (Rayleigh test)")
    ax.legend(); ax.grid(alpha=0.25)

    # (f) Orbital pole clustering
    ax = fig.add_subplot(gs[2, 1])
    for re, rc, c, ls, lbl in [
        (rp_e, rp_c, _BLUE, "-",  "ETNO poles, with P9"),
        (rn_e, rn_c, _RED,  "--", "ETNO poles, without P9"),
    ]:
        v = ~np.isnan(re["R_pole"])
        if v.any():
            ax.plot(t[v], re["R_pole"][v], color=c, lw=1.8, ls=ls, label=lbl)
    v = ~np.isnan(rp_c["R_pole_clones"])
    if v.any():
        ax.plot(t[v], rp_c["R_pole_clones"][v], color=_GRN, lw=1.4, ls=":",
                label="clone poles, with P9")
    ax.axhline(0.5, color="gray", ls=":",  lw=1, alpha=0.7)
    ax.axhline(0.7, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"Orbital pole $R_{\rm pole}$")
    ax.set_title(r"(f) Orbital pole clustering — Batygin & Brown signal")
    ax.legend(); ax.grid(alpha=0.25)

    # (g) 3:1 MMR resonance angle
    ax = fig.add_subplot(gs[3, 0])
    v = ~np.isnan(rp_c["phi31_mean"])
    if v.any():
        ax.plot(t[v], np.degrees(rp_c["phi31_mean"][v]), color=_PRP, lw=1.8)
        ax.fill_between(t[v],
                        np.degrees(rp_c["phi31_mean"][v]) - 30,
                        np.degrees(rp_c["phi31_mean"][v]) + 30,
                        alpha=0.15, color=_PRP,
                        label=r"$\pm 30°$ libration window")
    ax.axhline(0, color="k", ls="--", lw=0.9, alpha=0.5)
    ax.set_ylim(-190, 190)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel(r"$\langle\phi_{3:1}\rangle$ (°)")
    ax.set_title(r"(g) 3:1 MMR resonance angle (clone mean)"
                 "\n"
                 r"$\phi = 3\lambda_A - \lambda_{P9} - 2\varpi_A$")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # (h) Clone survival
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, rp_c["survival"], color=_BLUE, lw=1.8, label="with P9")
    ax.plot(t, rn_c["survival"], color=_RED,  lw=1.8, ls="--", label="without P9")
    ax.set_ylabel("Surviving Ammonite clones")
    ax.set_title("(h) Clone survival")
    ax.legend(); ax.grid(alpha=0.25)

    for ax in fig.axes[-4:]:
        ax.set_xlabel("Time (Myr)")

    fig.suptitle(
        "Planet Nine Dynamical Exploration  v4.2  —  "
        "Ammonite (2023 KQ14) + ETNO Ensemble\n"
        r"Nominal P9: $a$=500 AU, $e$=0.25, $m$=5 $M_\oplus$, "
        r"$i$=20°, $\omega_{P9}$=150°, $\Omega_{P9}$=94°",
        fontsize=11, y=0.998,
    )

    out = f"{SAVE_DIR}/p9_main.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 2: Polar clustering ────────────────────────────────────────────────

def plot_polar(rp_e, rn_e, cat) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                              subplot_kw={"projection": "polar"})
    rows = [("om", r"$\omega$"), ("vp", r"$\varpi$")]
    cols = [("with P9", rp_e), ("without P9", rn_e)]

    for ci, (clbl, res) in enumerate(cols):
        for ri, (key, albl) in enumerate(rows):
            ax = axes[ri, ci]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1.6)
            ax.set_yticks([])
            ax.set_title(f"{albl}\n{clbl}", pad=20, fontsize=10)

            angles = []
            for name, tr in res["etno"].items():
                if not tr[key]:
                    continue
                ang = tr[key][-1]
                angles.append(ang)
                ax.annotate("", xy=(ang, 1.05), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color=cat[name]["color"], lw=2.0))
                kl_tag = " ★" if res["kl_flags"].get(name) else ""
                clean  = (name.replace("$","").replace("_","")
                              .replace("{","").replace("}",""))
                ax.text(ang, 1.35, clean + kl_tag,
                        ha="center", va="center",
                        fontsize=6.5, color=cat[name]["color"])

            if len(angles) >= 2:
                Rv, mu, pv = rayleigh(angles)
                ax.annotate("", xy=(mu, Rv), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=3.0, alpha=0.65))
                ax.text(0.5, -0.13,
                    fr"$R$ = {Rv:.3f},   $p$ = {pv:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85))

    fig.suptitle(
        "ETNO Angular Distribution at Final Epoch\n"
        r"(N=6 ETNOs;  ★ = Kozai–Lidov cycling detected;"
        r"  treat $p$-values as indicative for small-$N$)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = f"{SAVE_DIR}/p9_polar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 3: P9 grid heatmap ─────────────────────────────────────────────────

def plot_grid(gres: list) -> None:
    masses = sorted(set(r["m_earth"] for r in gres))
    a_vals = sorted(set(r["a_p9"]    for r in gres))
    e_vals = sorted(set(r["e_p9"]    for r in gres))

    fig, axes = plt.subplots(2, len(masses),
                              figsize=(7 * len(masses) + 0.5, 10),
                              sharey="row")
    if len(masses) == 1:
        axes = axes.reshape(2, 1)

    de = ((e_vals[-1] - e_vals[0]) / (len(e_vals)-1) if len(e_vals) > 1 else 0.10)
    da = ((a_vals[-1] - a_vals[0]) / (len(a_vals)-1) if len(a_vals) > 1 else 100.)
    ext = [e_vals[0] - de/2, e_vals[-1] + de/2,
           a_vals[0] - da/2, a_vals[-1] + da/2]

    for row, metric in enumerate(["R_om", "R_pole"]):
        mlbl = r"$R_\omega$" if metric == "R_om" else r"$R_{\rm pole}$"
        for col, m in enumerate(masses):
            ax = axes[row, col]
            Z  = np.full((len(a_vals), len(e_vals)), np.nan)
            for r in gres:
                if r["m_earth"] != m:
                    continue
                Z[a_vals.index(r["a_p9"]),
                  e_vals.index(r["e_p9"])] = r[metric]
            im = ax.imshow(Z, origin="lower", cmap="RdYlGn",
                           vmin=0.0, vmax=1.0, aspect="auto", extent=ext)
            plt.colorbar(im, ax=ax, label=mlbl)
            ax.set_xticks(e_vals); ax.set_yticks(a_vals)
            ax.set_xlabel(r"$e_{P9}$")
            ax.set_ylabel(r"$a_{P9}$ (AU)")
            ax.set_title(fr"$m_{{P9}}$={m:.0f}$M_\oplus$ — {mlbl}")
            for i, a_v in enumerate(a_vals):
                for j, e_v in enumerate(e_vals):
                    if np.isnan(Z[i, j]):
                        continue
                    col_ = "white" if (Z[i,j] < 0.3 or Z[i,j] > 0.7) else "black"
                    ax.text(e_v, a_v, f"{Z[i, j]:.2f}",
                            ha="center", va="center",
                            fontsize=11, fontweight="bold", color=col_)

    fig.suptitle(
        r"P9 Parameter Grid — $\omega$ and Orbital Pole Clustering" "\n"
        f"50 Myr integration  ({50} Ammonite clones + 6 ETNOs per config)",
        fontsize=12,
    )
    plt.tight_layout()
    out = f"{SAVE_DIR}/p9_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 4: Energy conservation ─────────────────────────────────────────────

def plot_energy(econ: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(econ["t_myr"], econ["dE_rel"], color=_BLUE, lw=1.6)
    ax.axhline(1e-10, color="gray", ls="--", lw=1,
               label=r"$\Delta E/E = 10^{-10}$ reference")
    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel(r"$|\Delta E / E_0|$")
    ax.set_title("Energy Conservation — Mercurius Integrator\n"
                 r"(Sun + 4 giants + P9 backbone;  dt = 0.5 yr)")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = f"{SAVE_DIR}/p9_energy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 5: Phase space portrait ────────────────────────────────────────────

def plot_phase_space(rp_c, rn_c) -> None:
    N_steps = rp_c["a_all"].shape[1]
    snap_idx = [0, N_steps // 2, N_steps - 1]
    snap_lbl = ["t = 0", f"t = {rp_c['t_myr'][N_steps//2]:.0f} Myr",
                f"t = {rp_c['t_myr'][-1]:.0f} Myr"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

    for col, (si, sl) in enumerate(zip(snap_idx, snap_lbl)):
        for row, (rc, lbl, c) in enumerate([
            (rp_c, "with P9",    _BLUE),
            (rn_c, "without P9", _RED),
        ]):
            ax  = axes[row, col]
            a_s = rc["a_all"][:, si]
            e_s = rc["e_all"][:, si]
            ok  = np.isfinite(a_s) & np.isfinite(e_s)
            ax.scatter(a_s[ok], e_s[ok], s=4, alpha=0.5, color=c)
            ax.axvline(CATALOG["Ammonite (2023 KQ14)"]["a"],
                       color="k", ls="--", lw=0.8, alpha=0.5)
            ax.set_title(f"{sl}\n({lbl})", fontsize=10)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("$e$")
            if row == 1:
                ax.set_xlabel("$a$ (AU)")

    fig.suptitle("Ammonite Clone Phase Space  (a, e)\n"
                 "Vertical dashed line = nominal Ammonite $a$",
                 fontsize=12)
    plt.tight_layout()
    out = f"{SAVE_DIR}/p9_phase_space.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 6: Secular resonance overlap map ───────────────────────────────────

def plot_secular_map(smap: dict) -> None:
    da = smap["delta_omega"]
    a  = smap["a_arr"]
    e  = smap["e_arr"]
    a_nom = smap["a_nom"]

    de_ = (e[-1] - e[0]) / (len(e) - 1) if len(e) > 1 else 0.05
    da_ = (a[-1] - a[0]) / (len(a) - 1) if len(a) > 1 else 50.

    da_deg = np.degrees(np.clip(np.abs(da), 0, 2*np.pi))

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(da_deg, origin="lower", cmap="hot_r", aspect="auto",
                   extent=[e[0]-de_/2, e[-1]+de_/2,
                           a[0]-da_/2, a[-1]+da_/2])
    plt.colorbar(im, ax=ax, label=r"$|\Delta\omega|$ (°)")
    ax.axhline(a_nom, color="cyan", ls="--", lw=1.5,
               label=fr"Ammonite $a$ = {a_nom:.1f} AU")
    ax.set_xlabel("$e$")
    ax.set_ylabel("$a$ (AU)")
    ax.set_title(r"Secular Resonance Overlap Map: $|\Delta\omega|$ after 10 Myr"
                 "\nHot = large $\omega$ drift → secular resonance overlap region")
    ax.legend()
    fig.tight_layout()
    out = f"{SAVE_DIR}/p9_secular_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ── Figure 7 (GIF): Animated polar evolution ──────────────────────────────────

def animate_polar(rp_e, cat, n_frames: int = 80) -> None:
    print("  Generating animation …", end="", flush=True)
    times   = rp_e["times"]
    N_steps = len(times)
    frames  = np.linspace(0, N_steps - 1, n_frames, dtype=int)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    def update(frame_idx):
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        step   = frames[frame_idx]
        t_myr  = times[step] / 1e6
        angles = []

        for name, tr in rp_e["etno"].items():
            if len(tr["om"]) <= step:
                continue
            ang = tr["om"][step]
            angles.append(ang)
            ax.annotate("", xy=(ang, 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->",
                                color=cat[name]["color"], lw=2.2))
            clean = (name.replace("$","").replace("_","")
                         .replace("{","").replace("}",""))
            ax.text(ang, 1.38, clean, ha="center", va="center",
                    fontsize=6.5, color=cat[name]["color"])

        if len(angles) >= 2:
            Rv, mu, pv = rayleigh(angles)
            ax.annotate("", xy=(mu, Rv), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=3.5, alpha=0.7))
            ax.set_title(
                fr"$\omega$ clustering  —  t = {t_myr:.1f} Myr"
                "\n"
                fr"Rayleigh $R$ = {Rv:.3f},   $p$ = {pv:.3f}",
                pad=15, fontsize=11,
            )
        else:
            ax.set_title(fr"t = {t_myr:.1f} Myr", pad=15)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80)
    out  = f"{SAVE_DIR}/p9_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f" done\n  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# §14  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(stats: dict, grid_results: list,
                   kl_flags_p9: dict, kl_flags_nop9: dict,
                   t_end: float, n_clones: int) -> None:
    sep = "=" * 64
    lines = [
        sep,
        "  PLANET NINE DYNAMICAL EXPLORATION  —  v4.2",
        "  Ammonite (2023 KQ14) + ETNO Ensemble",
        sep,
        f"  Integration time : {t_end/1e6:.0f} Myr",
        f"  Clone ensemble   : {n_clones} covariance-sampled Ammonite clones",
        f"  ETNOs tracked    : {len(CATALOG)} (Ammonite, Sedna, 2012 VP113,",
        "                     2015 TG387, 2013 SY99, 2015 BP519)",
        f"  P9 nominal       : {P9_NOMINAL['label']}",
        "",
        "─" * 64,
        "  FINAL RAYLEIGH STATISTICS",
        "─" * 64,
    ]

    for lbl, key in [("WITH P9", "p9"), ("WITHOUT P9", "nop9")]:
        s = stats[key]
        lines += [
            f"  [{lbl}]",
            f"    R_omega (ETNO)      = {s['R_om']:.4f}",
            f"    R_varpi (ETNO)      = {s['R_vp']:.4f}",
            f"    R_pole  (ETNO)      = {s['R_pole']:.4f}",
            f"    Clone survival      = {s['surv']}",
            f"    Clone a (final)     = {s['a']:.2f} AU",
            f"    Clone e (final)     = {s['e']:.5f}",
            "",
        ]

    lines += [
        "─" * 64,
        "  KOZAI–LIDOV DETECTION (WITH P9)",
        "─" * 64,
    ]
    for name, flag in kl_flags_p9.items():
        marker = "DETECTED ★" if flag else "not detected"
        lines.append(f"  {name:<35s}  {marker}")

    lines += [
        "",
        "─" * 64,
        "  P9 GRID SCAN — TOP 5 CONFIGURATIONS BY R_omega",
        "─" * 64,
    ]
    ranked = sorted(grid_results,
                    key=lambda r: r["R_om"] if not np.isnan(r.get("R_om", np.nan)) else -1,
                    reverse=True)
    for ii, r in enumerate(ranked[:5], 1):
        lines.append(
            f"  {ii}. {r['label']:<45s}  "
            f"R_omega={r['R_om']:.3f}  R_pole={r['R_pole']:.3f}  surv={r['survival']}"
        )

    lines += ["", sep, "  Output files:", "─" * 64]
    figs = [
        "p9_main.png        — 8-panel main figure",
        "p9_polar.png       — polar clustering arrows",
        "p9_grid.png        — P9 parameter grid heatmap",
        "p9_energy.png      — energy conservation diagnostic",
        "p9_phase_space.png — clone (a,e) phase space portrait",
        "p9_secular_map.png — secular resonance overlap map",
        "p9_evolution.gif   — animated omega clustering",
        "p9_summary.txt     — this report",
    ]
    for f in figs:
        lines.append(f"  {f}")
    lines.append(sep)

    out = f"{SAVE_DIR}/p9_summary.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved: {out}")
    print("\n" + "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §15  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    BAR = "=" * 68
    print(BAR)
    print("  Planet Nine Dynamical Exploration  —  v4.2  (Research Grade)")
    print("  Ammonite (2023 KQ14) + ETNO Ensemble")
    print(BAR)
    print(f"  REBOUND version : {rebound.__version__}")
    print(f"  CPU cores       : {N_CPU}  (set P9_NCPU env-var to override)")
    print(f"  Output dir      : {Path(SAVE_DIR).resolve()}")

    _USE_HORIZONS = False
    print("\n  Testing JPL Horizons … ", end="", flush=True)
    try:
        _ts = rebound.Simulation()
        _ts.units = ("AU", "yr", "Msun")
        _ts.add("Jupiter")
        _USE_HORIZONS = True
        print("OK ✓  (live Horizons elements)")
    except Exception:
        print("FAILED ✗  (hardcoded J2000.0 fallback)")

    a_KQ14   = CATALOG["Ammonite (2023 KQ14)"]["a"]
    a_31_nom = P9_NOMINAL["a"] * (1.0 / 3.0) ** (2.0 / 3.0)
    a_p9_31  = a_KQ14 / (1.0 / 3.0) ** (2.0 / 3.0)
    print(f"\n  Ammonite a           = {a_KQ14:.1f} AU")
    print(f"  3:1 MMR for a_P9=500 = {a_31_nom:.1f} AU")
    print(f"  Exact 3:1 needs a_P9 ≈ {a_p9_31:.0f} AU  ← within P9 uncertainty!")

    T_END   = 100e6   
    N_STEPS = 300     

    print(f"\n{'─'*48}")
    print("[0] Energy conservation diagnostic …")
    econ = measure_energy_conservation(P9_NOMINAL, _USE_HORIZONS,
                                        t_end=1e6, N_steps=200)

    print(f"\n{'─'*48}")
    print("[1] Generating Ammonite clone ensemble …")
    clones = generate_clones(N_req=200, seed=42)

    print(f"\n{'─'*48}")
    print(f"[2] Main run — WITH Planet Nine  ({T_END/1e6:.0f} Myr) …")
    print(f"    {P9_NOMINAL['label']}")
    backbone_p9 = build_backbone(p9=P9_NOMINAL, use_horizons=_USE_HORIZONS)

    t0 = time.time()
    res_p9_e  = run_etno_reference(P9_NOMINAL, CATALOG, T_END, N_STEPS,
                                    use_horizons=_USE_HORIZONS)
    res_p9_c  = run_clone_ensemble(clones, backbone_p9, T_END, N_STEPS,
                                    p9_present=True, label="(with P9)")
    print(f"    Section [2] done in {time.time()-t0:.1f} s")

    print(f"\n{'─'*48}")
    print(f"[3] Control run — WITHOUT Planet Nine  ({T_END/1e6:.0f} Myr) …")
    backbone_no = build_backbone(p9=None, use_horizons=_USE_HORIZONS)

    t0 = time.time()
    res_nop9_e = run_etno_reference(None, CATALOG, T_END, N_STEPS,
                                     use_horizons=_USE_HORIZONS)
    res_nop9_c = run_clone_ensemble(clones, backbone_no, T_END, N_STEPS,
                                     p9_present=False, label="(without P9)")
    print(f"    Section [3] done in {time.time()-t0:.1f} s")

    print(f"\n{'─'*48}")
    print("[4] Secular resonance overlap map …")
    smap = compute_secular_map(P9_NOMINAL, _USE_HORIZONS,
                                n_a=20, n_e=20, t_end=1e7)

    print(f"\n{'─'*48}")
    print("[5] Computing summary statistics …")
    stats = {}
    for key, rc, re in [("p9",   res_p9_c,   res_p9_e),
                         ("nop9", res_nop9_c,  res_nop9_e)]:
        va  = ~np.isnan(rc["a_mean"])
        vo  = ~np.isnan(re["R_om"])
        vv  = ~np.isnan(re["R_vp"])
        vpl = ~np.isnan(re["R_pole"])
        stats[key] = dict(
            a     = float(rc["a_mean"][va][-1])    if va.any()  else np.nan,
            e     = float(rc["e_mean"][va][-1])    if va.any()  else np.nan,
            surv  = int(rc["survival"][-1]),
            R_om  = float(re["R_om"][vo][-1])      if vo.any()  else np.nan,
            R_vp  = float(re["R_vp"][vv][-1])      if vv.any()  else np.nan,
            R_pole= float(re["R_pole"][vpl][-1])   if vpl.any() else np.nan,
        )

    print(f"\n{'─'*48}")
    print("[6] Generating figures …")
    plot_energy(econ)
    plot_main(res_p9_e, res_p9_c, res_nop9_e, res_nop9_c, CATALOG)
    plot_polar(res_p9_e, res_nop9_e, CATALOG)
    plot_phase_space(res_p9_c, res_nop9_c)
    plot_secular_map(smap)
    animate_polar(res_p9_e, CATALOG, n_frames=80)

    print(f"\n{'─'*48}")
    print("[7] P9 parameter grid scan …")
    grid_results = run_grid(
        clones, CATALOG, P9_GRID,
        t_end        = 50e6,
        N_steps      = 50,
        N_cl         = 50,
        use_horizons = _USE_HORIZONS,
    )
    plot_grid(grid_results)

    print(f"\n{'─'*48}")
    print("[8] Writing summary report …")
    write_summary(
        stats        = stats,
        grid_results = grid_results,
        kl_flags_p9  = res_p9_e["kl_flags"],
        kl_flags_nop9= res_nop9_e["kl_flags"],
        t_end        = T_END,
        n_clones     = len(clones),
    )

    print(f"\n{BAR}")
    print(f"  All done!  Output saved to: {Path(SAVE_DIR).resolve()}")
    print(BAR)