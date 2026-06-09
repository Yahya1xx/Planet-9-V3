# Planet 9 Dynamical Exploration (v4.2)
### Research Grade Integration Pipeline — Ammonite (2023 KQ14) + Known ETNO Ensemble

---

## Scientific Motivation
This repository simulates the dynamical footprint of the hypothesized **Planet Nine**. Following the framework proposed by Batygin & Brown (2016, 2019, 2021), a $\sim$5–10 $M_{\oplus}$ planet situated between $\sim$400–800 AU is modeled to explain the observed clustering of extreme trans-Neptunian object (ETNO) orbital poles and arguments of perihelion. 

A primary focus of this version is **Ammonite (2023 KQ14)** (Wang et al. 2024, *Nature Astronomy*), which stands as the largest known ETNO ($H_V \sim 3.5$, $D \sim 600\text{ km}$), providing a vital new data point for testing these dynamical constraints.

---

## What's New in v4.2
Compared to v4.1, this release significantly optimizes integration efficiency and stability:
* **Integrator Upgrade:** Switched the baseline solver from `WHFast` to `Mercurius`.
* **Native Adaptive Timestepping:** Removed inefficient manual Python-level distance checks and loops. `Mercurius` now natively handles close encounters using the `IAS15` solver while maintaining a fixed base time step.
* **Performance Boost:** Drastically reduced execution overhead by delegating close-approach handling directly to the underlying C-engine.

---

## Integrator Architecture

### Giant-Planet Backbone
* **Engine:** `Mercurius` integrator (Rein et al. 2019)
* **Base Timestep ($dt$):** $0.5\text{ yr}$ ($\approx P_{\text{Jupiter}} / 24$)
* **Behavior:** Seamlessly transitions to `IAS15` during close planetary encounters, ensuring optimal energy conservation without manual sub-stepping.

### High-Eccentricity Test Particles
* **Targets:** Highly eccentric ETNOs such as Sedna ($e = 0.84$) and Goblin ($e = 0.94$).
* **Behavior:** `IAS15` step-size control automatically takes over during perihelion passages to preserve high tracking precision during deep gravitational plunges.

---

## REBOUND 5.0.0 API Implementation Reference

The following core implementation patterns are verified against the active environment:

### 1. Setup and Integrator Selection
```python
sim.integrator = "mercurius"
sim.dt = 0.5  # Base timestep handles close approaches automatically
