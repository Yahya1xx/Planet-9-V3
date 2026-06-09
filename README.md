
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

<img width="1665" height="1826" alt="p9_polar" src="https://github.com/user-attachments/assets/88ffeef7-c9d7-46c8-8a2c-bee57281b697" />
<img width="2380" height="1328" alt="p9_phase_space" src="https://github.com/user-attachments/assets/451e948f-17fc-4e34-b9f6-b6b72b544f54" />
<img width="1880" height="2621" alt="p9_main" src="https://github.com/user-attachments/assets/4b349fcd-8f45-49b1-9db9-d3d055e7d605" />
<img width="2118" height="1475" alt="p9_grid" src="https://github.com/user-attachments/assets/a947c1e1-0295-42ac-8ee8-74811d6585ae" />
<img width="1050" height="1050" alt="p9_evolution" src="https://github.com/user-attachments/assets/7c4f3133-fc61-40af-8de1-4b7e50892f83" />
<img width="1330" height="580" alt="p9_energy" src="https://github.com/user-attachments/assets/ac57019b-5a5d-4565-b886-ce93bde00332" />
<img width="1273" height="880" alt="p9_secular_map" src="https://github.com/user-attachments/assets/2ee72ff2-14e0-4779-b1bd-e5d7063dbbb1" />

