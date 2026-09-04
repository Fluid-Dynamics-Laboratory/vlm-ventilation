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
