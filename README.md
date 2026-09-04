# lifting_line_method-cavity_flow
A lifting-line method implementation to model the effects of dominant physics acting on a surface-piercing strut in ventilated ﬂow

## Running

Install the dependencies and open the notebooks from the repository root:

```
pip install -r requirements.txt
jupyter lab vlm_unsteady_wake.ipynb
```

The solver lives in `src/` (`VLMPanel`, `VLMSurface`, `VLMSolver`). It can be imported either as a package (`from src.VLMSolver import VLMSolver`) or with `src/` on `sys.path`, as done in the notebooks.
