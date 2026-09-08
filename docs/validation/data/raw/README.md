# Raw digitised data

Verbatim copies of the `data/` folder of L. Orain's repository
`lilianorain-oss/vortex_lattice_method-cavity_flow`, branch `wake_rel`, commit `4b91c3b`
(7 September 2026). Each file is one digitised curve: two columns `x; y` with a decimal comma,
and a first line starting with `#` that names the source. Several `#` lines were copied from one
file to the next and do not describe their file; the file name is authoritative, as follows.

| File | Source | x | y | Conditions |
|---|---|---|---|---|
| `cl_ar{AR}_fn{Fn}_w.csv` | Harwood, Young and Ceccio (2016), towing tank, fully wetted | incidence [deg] | $C_L$ on the immersed area $h c$ | `ar05`, `ar1`, `ar15`: $AR_h$ = 0.5, 1, 1.5; `fn15` = $Fn_h$ 1.5, `fn3` = 3.0, etc. |
| `cl_ar{AR}_fn{Fn}_v.csv` | same, fully ventilated | incidence [deg] | $C_L$ | same |
| `cavity.csv` | Harwood et al. (2016), photograph of the cavity | cavity length $L/c$ | depth $z/h$ | $AR_h$ = 1, $\alpha$ = 10°, $Fn_h$ = 1.5 |
| `Lc.csv`, `a0.csv`, `alpha.csv`, `cl.csv`, `sigma.csv` | Harwood et al. (2016), their lifting-line model, ventilated | $L/c$, $a_0/2\pi$, $\alpha_\text{eff}/\alpha$, $C_l$, $\sigma_c$ | depth $z/h$ | $AR_h$ = 1, $\alpha$ = 10°, $Fn_h$ = 1.5 |
| `lifting_FW.csv`, `lifting_FV.csv` | Harwood et al. (2016), their lifting-line model with the antisymmetric image | incidence [deg] | $C_L$ | $AR_h$ = 1; the Froude number was not recorded, the model curves in the same set are at $Fn_h$ = 1.5 |
| `rodriguez_vlm_1.csv`, `rodriguez_vlm_3.csv` | Rodriguez (1990), free-wake vortex lattice method | incidence [deg] | $C_L$ | rectangular wing, $AR$ = 1 and 3, no free surface |
| `lamar_exp_1.csv`, `lamar_exp_3.csv` | experiment reported by Lamar (1974) | incidence [deg] | $C_L$ | rectangular wing, $AR$ = 1 and 3 |
| `Tornado_vlm.csv` | Bertin and Smith (1998) analytic vortex lattice and Tornado (Melin, 2000), straight wake | incidence [deg] | $C_L$ | 45° swept rectangular wing, $AR$ = 5 |
| `experiment_weber.csv` | Weber and Brebner (1958), wind tunnel | incidence [deg] | $C_L$ | 45° swept rectangular wing, $AR$ = 5 |

The curves of `cavity.csv` and `Lc.csv` stop at $L/c$ = 2, the edge of the photograph. The
$C_L$ curves of the lifting-line model are attributed to $Fn_h$ = 1.5 in the comparison and
flagged as such. `import_digitised.py` in the parent folder converts these files into the
template format of `docs/validation/data/TEMPLATE.csv`.
