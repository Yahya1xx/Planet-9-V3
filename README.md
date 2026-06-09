# Planet-9-V3

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
