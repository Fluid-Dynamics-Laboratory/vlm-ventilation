# Ventilation inception model (branch `vent/inception`)

This branch adds a prediction of **whether, where and by which route** atmospheric
ventilation incepts on a lifting surface computed with the vortex lattice method. It
changes nothing in the solver: it reads the converged solution of a run and returns a
per-section assessment. The load model once ventilated is the subject of the companion
branch `vent/cavity-regime`; the two are independent and meet through a single predicate
(Sect. 6).

Files

| File | Content |
|---|---|
| `src/VLMInception.py` | the model: condition A per section, three routes for condition B, `assess`, `report` |
| `src/section_data.py` | two-dimensional section data (separation incidence, minimum pressure) read from a table |
| `src/section_data/naca0009.json` | provisional table for a NACA 0009 section |
| `docs/inception/make_naca0009_table.py` | builds that table and documents its sources |
| `docs/inception/validate.py`, `validate.json` | the validation cases of Sect. 5 and their results |
| `tests/test_inception.py` | unit checks of the section data and of the three routes |

## 1 Physical basis

Natural ventilation needs two things at once (Harwood, Young and Ceccio, J. Fluid Mech.
800, 2016; Young et al., Appl. Mech. Rev. 69, 2017): a region of separated flow at
sub-atmospheric pressure on the suction side, and a path for air from the free surface into
it. The path is opened by one of the trigger mechanisms catalogued in those papers and in
Aguiar Ferreira et al. (J. Fluid Mech. 1028, 2026): nose ventilation from leading-edge
separation at the waterline, tail ventilation from the depression behind a blunt trailing
edge, and tip-vortex ventilation when a tip vortex reaches the free surface. Perturbation
triggers (spray, waves, debris) are outside a steady model.

## 2 Condition A on each section

For a section whose control points are at depth $z$ below the free surface, with
$u_\infty$ the speed and $c$ the local chord,

$$\sigma_c(z) = \frac{2 g |z|}{u_\infty^2}, \qquad
  \text{separated} : \alpha_\text{gate} \ge \alpha_\text{sep}(Re_c), \qquad
  \text{sub-atmospheric} : -C_{p,\min}(\alpha_\text{eff}) \ge \sigma_c(z),$$

and the section is *ventilation-prone* when both hold. The effective incidence is the one
the nonlinear coupling of the code already uses,
$\alpha_\text{eff} = C_l / (2\pi \cos\lambda)$, from the sectional lift of the linearised
Kutta–Joukowski theorem, with $\lambda$ the mid-chord sweep. The separation incidence and the
minimum pressure coefficient come from the section table (Sect. 4). The free surface is the
plane $z = 0$ of the image.

Two gates are offered for the incidence that decides separation, because the two
experimental data sets disagree on it (Sect. 5):

- `seal="effective"` (default): the section separates at its own effective incidence. This
  is consistent with the sectional model. On a strut of immersed aspect ratio one with a
  pressure-release free surface, $\alpha_\text{eff}$ is about half the geometric incidence,
  so inception is predicted late.
- `seal="geometric"`: the section separates at the geometric incidence of the surface. This
  is the gate used by Viola's `bem-fem-fsi` and reproduces the incidence-only inception
  boundary of Harwood et al.

The sub-atmospheric test always uses the effective incidence.

## 3 Condition B, the three routes

**Nose.** A prone section lies within the waterline band, $|z| \le b\,c$. The band is the
depth over which the separated region is taken to reach the surface seal; with the
antisymmetric image the loading vanishes at the waterline itself. Default $b = 0.5$. On the
validation strut the predicted inception incidence is the same for $b$ = 0.25, 0.5 and 1.0
(Sect. 5), because the first section to become prone lies within a quarter chord of the
surface.

**Tail.** The trailing edge is blunt (`surface.blunt_te = True`), the base pressure is
sub-atmospheric in the waterline band, $-C_{p,\text{base}} \ge \sigma_c$, and the surface
depression can be amplified, $Fn_h > AR_h^{-1/2}$, the necessary condition derived by
Aguiar Ferreira et al. from the Rayleigh–Taylor growth of the depression. $Fn_h =
u_\infty/\sqrt{g h}$ and $AR_h = h/\bar c$ with $h$ the immersion.

**Tip vortex** (`submerged=True`, for an element that does not pierce the surface). The
vortex strength is the largest sectional circulation of the element, the core radius $r_c$
is the solver's Scully core, and the trajectory is that of the edge filaments of the free
wake. The route is open when the core is sub-atmospheric at the shallowest point of the
trajectory,

$$C_{p,\text{core}} = -\frac{\Gamma^2}{4\pi^2 r_c^2 u_\infty^2} \le -\sigma_c(z_\text{min}),$$

and that point lies within $k\,r_c$ of the free surface ($k = 1$ by default). **This route is
unvalidated**: no published experiment isolates tip-vortex ventilation of a submerged
element, and the result carries `validated: False`.

## 4 Section data, provisional

`src/section_data/naca0009.json` holds, for a NACA 0009 section:

- $C_{p,\min}(\alpha)$ for $\alpha$ = 0 to 25°, from a linear-strength vortex panel method
  (Kuethe and Chow, Sect. 5.10) on the closed-trailing-edge thickness form. It is
  **inviscid**, so it bounds from above the suction the real section can hold; the
  sub-atmospheric test is therefore conservative.
- $\alpha_\text{sep}(Re_c)$ for $Re_c$ = $10^5$ to $9\times10^6$, read to the nearest half
  degree as the incidence of maximum lift from Sheldahl and Klimas (Sandia report
  SAND80-2114, 1981) below $10^6$ and Abbott and von Doenhoff (Theory of Wing Sections, 1959)
  above $3\times10^6$.
- a base pressure coefficient of $-0.3$ for a blunt trailing edge (Hoerner, Fluid-Dynamic
  Drag, Ch. 3).

The table is marked `PROVISIONAL` in the file and in every report that uses it. It should be
replaced by XFOIL polars in the same format, which would give the separation location, the
transition and a viscous $C_{p,\min}$. `make_naca0009_table.py` regenerates it.

## 5 Validation against published scalars

All cases: `python docs/inception/validate.py` from the repository root, about 20 min.
Strut of Harwood et al.: NACA 0009-type section, chord 0.2794 m (assumed), $AR_h = 1$,
12 sections over the immersion, 5 chordwise panels, antisymmetric image, three chord
lengths of travel.

**Case 1, nose route, effective gate.**

| $Fn_h$ | $Re_c$ | $\alpha_\text{sep}(Re_c)$ | predicted inception |
|---|---|---|---|
| 1.5 | $6.9\times10^5$ | 11.0° | 23° |
| 2.0 | $9.3\times10^5$ | 11.8° | 24° |
| 2.5 | $1.2\times10^6$ | 12.2° | 24° |
| 3.5 | $1.6\times10^6$ | 12.7° | 25° |

Published: Aguiar Ferreira et al. (2026) measure spontaneous inception under steady
conditions at 20° to above 25° for $Fr$ 1.25 to 2.5 and $AR$ 1 to 1.5, rising with Froude
number, on a NACA 0010-34. The prediction lies in that range with the same trend, which here
comes from the Reynolds dependence of the separation incidence. The band parameter does not
move the result (0.25, 0.5 and 1.0 chord all give 24° at $Fn_h = 2.5$).

**Case 1c, nose route, geometric gate.**

| $Fn_h$ | predicted inception | Harwood et al. (2016) |
|---|---|---|
| 1.5 | 11.0° | ca. 14.5° |
| 2.5 | 12.5° | ca. 14.5° |
| 3.5 | 13.0° | ca. 14.5° |

The geometric gate reproduces an incidence-only boundary, as Harwood et al. observed. The
2 to 3° offset is the difference between the separation incidence of a plain NACA 0009 in
the provisional table and that of the modified section of the experiment, and it is the
first thing XFOIL polars of the actual section would correct. Harwood et al. formed their
cavities during acceleration and with perturbations, which is why their boundary sits at the
lower edge of the bistable region that Aguiar Ferreira et al. later mapped; the two gates
bracket the two data sets.

**Case 2, tail route** (blunt trailing edge, $\alpha = 10°$): closed at $Fn_h = 0.8$ and
open at 1.0, 1.5 and 2.5, the threshold being $AR_h^{-1/2} = 0.98$. The threshold is
implemented as published, so this case verifies the implementation, not the physics.

**Case 3, tip-vortex route** on a generic swept element ($AR = 6$, sweep 20°, $\alpha = 6°$,
6 m/s, chord 0.3 m): closed at depths of 0.1 to 1.0 chord. The shallowest point of the tip
filament stays 0.05 to 0.31 m below the surface against a core radius of 0.015 m, and the
core suction, $C_{p,\text{core}}$ = $-0.23$ to $-0.44$, would be sub-atmospheric only in
the top 0.4 to 0.8 m. The route therefore requires the vortex to rise to the surface, which
a free wake of four chords at this loading does not show. Unvalidated, reported only.

## 6 Interface to the cavity and regime branch

`VLMInception.assess(surface, solver, data, ...)` returns a dictionary with the per-section
state, the prone fraction of the span, the three routes and `incepts`. The regime machine of
`vent/cavity-regime` takes any callable `(surface, solver) -> assessment` with those keys, so
this module plugs in unchanged at merge.

## 7 Limitations

- The boundary-layer physics is in the table, not in the model. Reynolds and Weber effects
  at model scale (Damley-Strnad, Harwood and Young, smp'19, 2019) enter only through
  $\alpha_\text{sep}(Re)$.
- The waterline band and the tip-path factor are provisional parameters.
- The free surface is the high-Froude image; the model warns below $Fn_c = 1.5$.
- The tip-vortex route is unvalidated.
