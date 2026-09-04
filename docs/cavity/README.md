# Ventilated cavity and flow regimes (branch `vent/cavity-regime`)

This branch gives the vortex lattice method a **ventilated cavity whose effect on the loads is
solved on the lattice**, and a **hysteretic regime machine** that decides when a cavity forms,
persists and washes out. It replaces the prescribed sectional twist of the internship report
by a cavity at the pressure of an atmospheric cavity at each depth, with a thickness, and by
the formation, persistence and elimination rules of Harwood, Young and Ceccio (J. Fluid Mech.
800, 2016). The inception signal is an input: the companion branch `vent/inception` supplies
it, and a stall-angle gate is provided here so that the branch runs alone.

Files

| File | Content |
|---|---|
| `src/VLMCavity.py` | `CavitySolver`, the source-panel kernel, the mixed solve, the sectional extent, the two branches, the regime machine, `stall_gate`, `carry_state` |
| `src/vent_section.py` | closed-form sectional relations, reused verbatim from I. M. Viola's `bem-fem-fsi` (MIT licence): Acosta (1955), Tulin (1953), the fits of Harwood et al., closure angle, washout boundary |
| `src/VLMSolver.py` | two hooks in the time stepping (`_solve_step`, `_after_step`); the cutoff reference length corrected (Sect. 7) |
| `docs/cavity/verify_acosta.py`, `.json` | source kernel checks and the two-dimensional verification against Acosta |
| `docs/cavity/validate.py`, `.json` | the strut of Harwood et al.: loads, closure angle, hysteresis |
| `tests/test_cavity.py` | unit checks |

## 1 Formulation

**Cavity on the lattice.** On a ventilated panel the ring vortex of the lattice is joined by a
constant-strength source sheet $q$, the transpiration that gives the cavity its thickness. Two
conditions hold on such a panel instead of the one kinematic condition of a wetted panel:

$$(\boldsymbol{u}_\infty + \boldsymbol{v}_\text{rings} + \boldsymbol{v}_\text{sources})\cdot\boldsymbol{n}_\text{p} = 0
\quad\text{on the pressure side,}$$

$$\boldsymbol{u}_\infty\cdot\boldsymbol{\tau} + \boldsymbol{v}_\text{wake}\cdot\boldsymbol{\tau}
 + \boldsymbol{v}_\text{sources}\cdot\boldsymbol{\tau} + \tfrac12\gamma_i = u_\infty\left(1 + \tfrac12\sigma_c\right)
\quad\text{on the cavity side,}$$

where $\gamma_i = (\Gamma_i - \Gamma_{i-1})/\Delta c_i$ is the bound vorticity per unit chord
of the panel, whose jump across the sheet is the only tangential velocity a planar lattice
induces on itself, and $\sigma_c(z) = 2 g |z|/u_\infty^2$ is the cavitation number of an
atmospheric cavity at the depth of the panel. The dynamic condition is linearised as in Acosta
(1955), so that the two-dimensional verification compares like with like. The source sheet
emits $q/2$ on each side; its normal velocity is evaluated at a point displaced to the pressure
side, so the own-panel term enters without a special case. The unknowns are the ring strengths
of all panels and the source strengths of the cavity panels; the system is square and is solved
directly at every time step.

**Source panel kernel.** Constant-strength quadrilateral source of Hess and Smith (1962). The
in-plane velocity uses the edge logarithms of Katz and Plotkin, Sect. 10.4.1; the normal
velocity is the solid angle of the panel at the point, $w = \Omega/4\pi$, which is exact for
edges parallel to either in-plane axis where the textbook formula has a removable singularity.
Checks: $w = \pm 0.4999990$ on the two sides of a unit square, far field within 0.01 % of a
point source, in-plane velocity of a long strip equal to the two-dimensional sheet value to
$10^{-4}$.

**Cavity extent.** The length $L_j$ of the cavity on section $j$ comes from the sectional
theory: $\Psi_j = \sigma_c(z_j)/(2\alpha_{\text{eff},j})$ with $\alpha_{\text{eff},j} =
C_{l,j}/2\pi$ from the wetted solution of the same step, and $L_j = L(\Psi_j)$ from the blend of
Acosta's partial cavity and Tulin's supercavity in `vent_section.cavity_length` (Harwood et al.
2016, Fig. 1.4 of the report). The cavity covers whole panels up to $L_j$ and the load is
interpolated between the two bracketing panel counts, so that it is continuous in the
incidence. Sect. 4 records why the length is not iterated on the lattice.

**Two branches per section.** The linear closed-cavity theory holds for $L \le 0.5$ (Acosta's
own limit). Up to that length the section is solved on the lattice as above and the lift
follows from the redistributed circulation. Beyond it the model truncated at the trailing edge
is not valid, and the sectional lift slope $a_0(L)$ of `vent_section.lift_slope` is imposed on
the section by scaling the free-stream term of its kinematic condition by $a_0/2\pi$, which is
the effective-incidence correction of the report in a conservative form. The two branches agree
to 4 % at $L = 0.5$ (Sect. 3). The length used for the loads is capped at 10 chords, where
$a_0$ is within 1 % of its supercavitating limit $\pi/2$.

**Loads.** Lift by the linearised Kutta–Joukowski theorem of the base solver. The centre of
pressure of each section, $e$ forward of mid-chord, is computed from the chordwise bound
vorticity on the lattice branch and from `vent_section.centre_of_pressure` on the sectional
branch.

## 2 Regime machine

Per surface: regime FW, PV or FV; cavity length per section; a growth rate limit (chords of
cavity per chord of travel, default 0.2) so that a cavity forms over a few convective times.

- **Formation** needs an inception signal: any callable `(surface, solver)` returning the
  dictionary of `VLMInception.assess` (keys `incepts`, `route`, `state["prone"]`), or
  `stall_gate(alpha)`, or `vent.force(surface)` for the perturbation route of the experiments.
  The cavity is seeded on the prone sections that can hold one (sectional length of at least
  one panel).
- **Persistence** asks only whether a cavity can still exist on the section. It never asks
  again for the seal, and that asymmetry is the hysteresis.
- **Elimination** follows the re-entrant jet as in Sect. 4 of Harwood et al.: at the
  representative mid-depth section the local angle of the closure line from the flow exceeds
  45°, that is $|dx_\text{closure}/dz| < 1$. The closure line is evaluated with the long-cavity
  relation $L = 2.31/\Psi$, the one Harwood et al. use to derive their washout boundary
  (Sect. 5 explains why). Above the formation boundary an unstable cavity persists and sheds
  (PV); below it the flow rewets after the cavity has been unstable for three chords of travel.
- **Label**: `vent_section.regime`: FW when no section is ventilated, FV when the cavity
  reaches the tip and the closure line is stable, PV otherwise.

Sweeps carry the state from one incidence to the next with `carry_state`.

## 3 Verification against Acosta (two-dimensional)

Rectangular wing of aspect ratio 40 at 4°, uniform prescribed cavitation number, 24 sections,
16 chordwise panels, eight chords of travel; mid-span section; the wetted reference at the same
travel gives $a_0 = 5.77$ (finite travel and span), and the ratios below cancel that.
`python docs/cavity/verify_acosta.py`.

| $\Psi$ | $L$ (sectional) | $L$ Acosta | $a_0/a_{0,\text{wet}}$ lattice | $a_0(L)/2\pi$ Acosta | branch |
|---|---|---|---|---|---|
| 12 | 0.103 | 0.111 | 1.074 | 1.028 | lattice |
| 8 | 0.250 | 0.253 | 1.130 | 1.077 | lattice |
| 6 | 0.500 | 0.467 | 1.250 | 1.207 | lattice |
| 4 | 0.850 | outside Acosta's range | 1.014 | 1.005 (fit) | sectional |
| 2 | 1.159 | outside Acosta's range | 0.771 | 0.766 (fit) | sectional |

The lattice branch reproduces Acosta's lift within 4 to 5 % over his whole range, including
the rise of the lift slope above $2\pi$ that a short cavity produces by its displacement. The
sectional branch reproduces the fit by construction. The centre of pressure of the lattice
branch moves forward with the cavity, $e$ = 0.24 wetted to 0.32 at $L = 0.5$, whereas the blend
of Harwood et al. moves it aft towards $3c/16$; the blend is a tanh interpolation between the
wetted and supercavitating limits and not Acosta's moment, so this difference is reported, not
resolved.

## 4 What was tried and not retained

A closure iteration on the lattice, the length of each section adjusted until the cavity
thickness $h(L) = \int_0^L q\,dx / u_\infty$ vanishes at its end, was implemented and tested.
With the length prescribed at Acosta's value the mixed solve returns the lift within 5 %, but
the source strengths alternate in sign from panel to panel, the classic odd–even decoupling of
a collocated source–vortex lattice, so the thickness at closure is noise and its zero crossing
lands at $L \approx 0.3$ instead of 0.47 for $\Psi = 6$. The potential-based (Dirichlet)
formulation of `bem-fem-fsi` does not have this problem, which is one reason to prefer it for
a single strut. Here the length is taken from the sectional theory and the closure residual is
kept as a diagnostic (`vent.state[...]["closure"]`).

## 5 Validation on the strut of Harwood et al.

Rectangular NACA 0009-type strut, chord 0.2794 m (assumed), $AR_h = 1$, atmospheric cavity,
antisymmetric image, 12 sections, 8 chordwise panels, three chords of travel (four in the
sweeps). `python docs/cavity/validate.py`, about 20 min.

**Case A, fully ventilated branch imposed, $Fn_h = 2.5$.** The ratio of ventilated to wetted
lift is compared with the semi-empirical relation of Damley-Strnad, Harwood and Young (smp'19,
2019, Eq. 7), fitted to the towing-tank data of several campaigns.

| $\alpha$ | $C_L$ wetted | $C_L$ ventilated | ratio, this model | ratio, Damley-Strnad et al. | $e$ wetted | $e$ ventilated | regime |
|---|---|---|---|---|---|---|---|
| 6° | 0.162 | 0.133 | 0.82 | 0.96 | 0.224 | 0.191 | PV |
| 10° | 0.299 | 0.190 | 0.64 | 0.69 | 0.203 | 0.188 | FV |
| 14° | 0.463 | 0.244 | 0.53 | 0.57 | 0.189 | 0.188 | FV |
| 18° | 0.661 | 0.296 | 0.45 | 0.53 | 0.174 | 0.188 | FV |

The trend and the magnitude follow the semi-empirical relation, this model being 5 to 15 %
lower; both lie in the 50 to 70 % loss reported for the fully ventilated regime at high
incidence, and both recover towards the wetted value at low incidence where the cavity
shortens. The panel method of `bem-fem-fsi` gives 0.54 to 0.58 at the same conditions. At
$Fn_h = 2.5$ every section is on the sectional branch ($L > 0.5$ everywhere), so the lattice
cavity branch does not enter these numbers; it enters at lower Froude number or lower
incidence, where the cavity is short. The ventilated centre of pressure is $3c/16$ by
construction of the blend on the sectional branch; the wetted value comes from the lattice.

**Case B, closure angle at $\alpha = 20°$.** Harwood et al. measured a mean closure angle of
40.75° at washout and proposed 45° as the limit.

| $Fn_h$ | closure angle at mid-depth, this model | $C_L$ | washout $Fn_h$ of Harwood's boundary at this $C_L$ |
|---|---|---|---|
| 1.5 | 34.5° | 0.506 | 1.10 |
| 2.0 | 22.5° | 0.401 | 1.24 |
| 2.5 | 15.4° | 0.322 | 1.38 |

At the published condition the model gives 34.5° against the measured 40.75°, 15 % low, and it
classes the cavity as stable there, consistent with Harwood's boundary, which places washout at
$Fn_h = 1.10$ for this lift. Evaluating the same line with the blended length relation instead
of the long-cavity one gives 61° at mid-depth, because the blend flattens for $1 < L < 3$; that
is why the stability criterion uses the relation Harwood et al. derived it with.

**Case C, hysteresis at $Fn_h = 2.5$** with the stall gate at 14.5°.

| sweep | 4° | 6° | 8° | 10° | 12° | 14° | 16° | 18° | 20° |
|---|---|---|---|---|---|---|---|---|---|
| up, $C_L$ | 0.102 | 0.162 | 0.228 | 0.299 | 0.378 | 0.463 | 0.270 | 0.296 | 0.322 |
| up, regime | FW | FW | FW | FW | FW | FW | FV | FV | FV |
| down, $C_L$ | 0.102 | 0.133 | 0.163 | 0.190 | 0.218 | 0.245 | 0.270 | 0.296 | 0.322 |
| down, regime | FW | PV | PV | FV | FV | FV | FV | FV | FV |

The ascending branch stays wetted to 14° and ventilates at 16°; the descending branch stays
fully ventilated to 10°, sheds partially at 8° and 6°, and rewets at 4° by the re-entrant-jet
criterion. The bistable band, 6° to 14°, is the band Harwood et al. report at this Froude
number (their Fig. 16) and the one `bem-fem-fsi` computes (4° to 14°). At 10° the two branches
carry $C_L$ = 0.299 and 0.190 at identical conditions; only the history differs.

## 6 Interface to the inception branch

`CavitySolver(..., inception=f)` accepts any `f(surface, solver)` returning the assessment of
`VLMInception.assess`. At merge the stall gate is replaced by that function and nothing else
changes.

## 7 A correction to the base solver carried on this branch

The cutoff distance of the influence matrix was a fraction of the smallest panel *width*. On a
lattice with wide panels, a plate of aspect ratio 40 with 24 sections and 16 chordwise panels,
the width exceeds the panel chord and the cutoff removed the neighbouring bound segments from
the matrix, whose condition number rose to $10^{13}$. The reference length is now the smallest
panel dimension, width or chord. The cases of the report, whose panels are narrower than their
chord, are unchanged ($C_L$ = 0.3863 for the aspect-ratio-one wing at 10°). The same commit
should be applied to the other branches.

## 8 Limitations

- The cavity length is sectional, not solved on the lattice (Sect. 4).
- The linear partial-cavity branch stops at $L = 0.5$; the long-cavity branch is the sectional
  fit, so the lattice adds nothing to the load of a supercavitating section beyond its
  spanwise coupling.
- The closure angle is 15 % low at the one published condition, and its aspect-ratio trend is
  untested here; `bem-fem-fsi` found the trend wrong in its own model too.
- Formation and washout time scales (growth rate 0.2 chords per chord, washout after three
  chords) are parameters, chosen from the 3.5 to 7 convective times of Aguiar Ferreira et al.
  (2026) and not fitted.
- Wetted drag, cavity drag and spray are not modelled.
