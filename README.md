# lifting_line_method-cavity_flow
A lifting-line method implementation to model the effects of dominant physics acting on a surface-piercing strut in ventilated ﬂow

## Running

Install the dependencies and open the notebooks from the repository root:

```
pip install -r requirements.txt
jupyter lab vlm_unsteady_wake.ipynb
```

The solver lives in `src/` (`VLMPanel`, `VLMSurface`, `VLMSolver`). It can be imported either as a package (`from src.VLMSolver import VLMSolver`) or with `src/` on `sys.path`, as done in the notebooks.

## Numerical parameters of the solver

`VLMSolver(surfaces, u_inf, boundary, ratio, a_ratio)`

- `a_ratio`: cutoff distance for the boundary condition (influence matrix and right-hand side), as a fraction of the smallest panel width. Keep it below 0.5 so that no control point lies inside the cutoff of its own ring. Default 0.4.
- `ratio`: vortex core radius (Scully profile) for the wake roll-up and the induced-velocity loads, as a fraction of the mean chord. Default 0.05; results vary by ca. 1 % between 0.02 and 0.1.

The rings shed at the current time step from the trailing edge and from the tips carry the current circulation of the panels they are shed from and are part of the influence matrix (implicit Kutta condition). The lift is evaluated with the linearised Kutta-Joukowski theorem; the induced-velocity contribution is kept apart in `surface.loads_induced`.

The study behind these choices is in `docs/tip_loading/`.

## Ventilation inception (branch `vent/inception`)

`src/VLMInception.py` assesses, after a run, whether atmospheric ventilation incepts on a
surface and by which route (nose, tail, tip vortex), from the two necessary conditions of
Harwood, Young and Ceccio (2016): separated sub-atmospheric flow on a section, and a path to
the free surface. Section data (separation incidence, minimum pressure, separation location and
a confidence value against incidence and Reynolds number) come by default from NeuralFoil, a
pure Python surrogate of XFOIL installed with pip, through `SectionData.from_neuralfoil`; any
table in the same JSON format, for instance XFOIL polars, can replace it. Model, parameters,
validation against published values and limitations: `docs/inception/README.md`.

```python
from section_data import SectionData
import VLMInception as vi
result = vi.assess(surface, vlm, SectionData.naca0009(), g=9.81, nu=1e-6)   # after vlm._kuttas_loads()
vi.report(result)
```
## Ventilated cavity and flow regimes (branch `vent/cavity-regime`)

`src/VLMCavity.py` adds to the solver a sheet cavity on the suction side of each section, at
the pressure of an atmospheric cavity at that depth, whose effect on the circulation and the
loads is solved on the lattice (linearised partial-cavity theory, verified against Acosta 1955),
and a hysteretic regime machine (fully wetted, partially and fully ventilated) with formation
by an inception signal, persistence, and elimination by the re-entrant jet criterion of
Harwood, Young and Ceccio (2016). `src/vent_section.py` is the closed-form sectional reference,
reused from I. M. Viola's bem-fem-fsi (MIT). Model, verification, validation and limitations:
`docs/cavity/README.md`.

```python
from VLMCavity import CavitySolver, stall_gate
vlm = CavitySolver([surface], np.array([U, 0, 0]), "antisymmetric", 0.05, 0.4, g=9.81, inception=stall_gate(14.5))
vlm._time_sim(T, DT, "classic"); vlm._kuttas_loads()
vlm.vent.report(surface)          # regime, cavity length and centre of pressure of every section
```

This branch also corrects the reference length of the cutoff in the base solver: it is now the
smallest panel dimension, width or chord, so that wide panels (high aspect ratio with few
sections) no longer exclude neighbouring segments from the influence matrix.
