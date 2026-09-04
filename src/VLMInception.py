"""
Ventilation inception on a lifting surface computed with the vortex lattice method.

The model follows the two necessary conditions of Harwood, Young & Ceccio (J. Fluid Mech.
800, 2016) and Young et al. (Appl. Mech. Rev. 69, 2017): natural ventilation needs (A) a
region of separated flow at sub-atmospheric pressure on the suction side and (B) a path for
air from the free surface into that region. Condition A is evaluated on every spanwise
section of the lattice from the sectional effective incidence and depth; condition B is
evaluated for three routes.

Condition A at a section of depth z below the free surface (z = 0 is the free surface, the
plane of the image; the depth is |z| of the section control points):

    sigma_c(z) = 2 g |z| / u_inf**2                      cavitation number of an atmospheric cavity
    separated  : alpha_eff >= alpha_sep(Re)               from the section data
    sub-atm.   : -Cp_min(alpha_eff) >= sigma_c(z)         the suction exceeds the hydrostatic head
    prone      = separated and sub-atmospheric

Routes for condition B:

    nose   : a prone section lies within the waterline band, |z| <= band*c. With the
             antisymmetric image the loading vanishes at the waterline, so the band is the
             depth over which the surface seal is taken to be reachable by the separated
             region. Default band = 0.5 chord (provisional parameter).
             Two gates are offered for the separation that breaks the seal (`seal`):
               "effective" : the section is separated at its effective incidence, the one
                             the lattice computes (consistent with the sectional model; on a
                             low-aspect-ratio strut alpha_eff is about half the geometric
                             incidence, so inception is predicted late, in line with the
                             steady-state map of Aguiar Ferreira et al. 2026);
               "geometric" : the section is separated at the geometric incidence of the
                             surface, as in Viola's bem-fem-fsi, which reproduces the
                             incidence-only inception boundary of Harwood et al. (2016).
             The sub-atmospheric test always uses the effective incidence.
    tail   : the trailing edge is blunt (surface.blunt_te), its base pressure is
             sub-atmospheric at the waterline band, -Cp_base >= sigma_c, and the surface
             depression can be amplified, Fn_h > AR_h**(-1/2) (Aguiar Ferreira et al.,
             J. Fluid Mech. 2026). Fn_h = u/sqrt(g h), AR_h = h/c_mean.
    tip    : the tip vortex of a submerged element is at sub-atmospheric core pressure,
             -Cp_core >= sigma_c(z_v), with Cp_core = -Gamma**2/(4 pi**2 r_c**2 u**2) for a
             Scully core of radius r_c, and its trajectory in the free wake comes within
             k_path*r_c of the free surface. UNVALIDATED: no published experiment isolates
             this route for a submerged element; it is reported with that label.

The effective incidence of a section is the one used by the nonlinear coupling of the code,
alpha_eff = Cl_2d / (a0 cos(lambda)), from the sectional lift of the linearised
Kutta-Joukowski theorem (Liguori et al. 2025 for the sweep correction), so the inception
model consumes exactly what the coupling algorithm already computes.

Free-surface validity: the antisymmetric image is the high-Froude limit; the model warns
when the chord Froude number u/sqrt(g c) is below 1.5, following the validation of the
internship report (Sect. 3.1.1).

Everything here is a pure function of the surfaces and the solver after a run; nothing is
mutated. `assess` returns a dictionary of arrays and flags, `report` prints it.
"""
import warnings
import numpy as np



def section_state(surface, solver, data, g=9.81, nu=1.0e-6, band=0.5, seal="effective"):
    """
    Per-section quantities and condition A for one surface after a solve with loads.

    Input
        surface : VLMSurface with gamma, Cl_2d and wing_panels set (after _kuttas_loads)
        solver  : VLMSolver (for u_inf and the core radius)
        data    : SectionData
        g, nu   : gravity [m/s^2] and kinematic viscosity [m^2/s]
        band    : waterline band, in chords, over which condition A opens the nose route
        seal    : "effective" or "geometric", the incidence that decides separation (see module docstring)
    Output
        dict of arrays over the N sections: y (spanwise coordinate along the surface),
        depth, chord, sigma_c, Re, alpha_eff [deg], alpha_sep [deg], cp_min, separated,
        sub_atmospheric, prone, in_band; and scalars u, h (immersion), c_mean, Fn_h, AR_h, Fn_c.
    """
    m, n = surface.M, surface.N
    u = float(np.linalg.norm(solver.U))
    panels = surface.wing_panels["real"]
    ctrl = np.array([p.ctr for p in panels]).reshape(m, n, 3)
    chord = np.array([m*np.linalg.norm(p.chord) for p in panels]).reshape(m, n)[0]     # local chord per section
    depth = np.abs(ctrl[:, :, 2]).mean(axis=0)                                           # |z| of the section
    span_coord = ctrl[:, :, 1 if surface.plan == 1 else 2].mean(axis=0)
    sigma_c = 2*g*depth/u**2
    re = u*chord/nu
    cl2d = np.abs(np.asarray(surface.Cl_2d))
    alpha_eff = np.rad2deg(cl2d/(data.a0*np.cos(surface.sweep)))
    alpha_sep = data.alpha_sep(re)
    cp_min = data.cp_min(alpha_eff)
    alpha_geo = np.rad2deg(abs(surface.aoa) + abs(surface.drift)) + np.abs(np.rad2deg(_twist(surface, n)))
    alpha_gate = alpha_eff if seal == "effective" else alpha_geo
    separated = alpha_gate >= alpha_sep
    sub_atm = -cp_min >= sigma_c
    prone = separated & sub_atm
    h = float(depth.max() + 0.5*(depth.max() - depth.min())/max(n - 1, 1))              # immersion, to the tip edge
    c_mean = float(chord.mean())
    fnh = u/np.sqrt(g*h); fnc = u/np.sqrt(g*c_mean)
    if fnc < 1.5:
        warnings.warn(f"chord Froude number {fnc:.2f} < 1.5: the antisymmetric free-surface image is outside its validated range")
    return dict(y=span_coord, depth=depth, chord=chord, sigma_c=sigma_c, Re=re, alpha_eff=alpha_eff,
                alpha_geo=alpha_geo, alpha_sep=alpha_sep, cp_min=cp_min, separated=separated,
                sub_atmospheric=sub_atm, prone=prone, in_band=depth <= band*chord, seal=seal,
                u=u, h=h, c_mean=c_mean, Fn_h=fnh, AR_h=h/c_mean, Fn_c=fnc)


def _twist(surface, n):
    """Geometric twist of each section [rad]; a scalar twist is the tip value of a linear distribution."""
    phi = surface.twist
    if isinstance(phi, np.ndarray):
        return np.asarray(phi, float)[:n] if len(phi) >= n else np.zeros(n)
    return np.linspace(0.0, float(phi), n)


def nose_route(state):
    """Nose ventilation: a prone section inside the waterline band. -> (bool, index of the seed section or None)"""
    idx = np.where(state["prone"] & state["in_band"])[0]
    return (len(idx) > 0), (int(idx[np.argmin(state["depth"][idx])]) if len(idx) else None)


def tail_route(state, surface, data):
    """Tail ventilation behind a blunt trailing edge. -> (bool, dict of the three sub-conditions)"""
    blunt = bool(getattr(surface, "blunt_te", False))
    band = state["in_band"]
    base_sub_atm = bool(np.any((-data.cp_base >= state["sigma_c"]) & band)) if blunt else False
    amplified = state["Fn_h"] > state["AR_h"]**-0.5
    return (blunt and base_sub_atm and amplified), dict(blunt_te=blunt, base_sub_atmospheric=base_sub_atm,
                                                        surface_amplified=bool(amplified),
                                                        Fn_h_threshold=float(state["AR_h"]**-0.5))


def tip_vortex_route(surface, solver, g=9.81, k_path=1.0):
    """
    Tip-vortex ventilation of a submerged element (UNVALIDATED).

    The vortex strength is the largest sectional circulation of the element, the core radius the
    solver's Scully core, and the trajectory the tip columns of the free wake (both tips). The
    route is open when the core is sub-atmospheric at the shallowest point of the trajectory and
    that point is within k_path core radii of the free surface.
    -> (bool, dict)
    """
    m, n = surface.M, surface.N
    u = float(np.linalg.norm(solver.U))
    gamma_te = np.abs(np.asarray(surface.gamma).reshape(m, n)[-1])       # circulation of each section
    gamma_v = float(gamma_te.max())
    rc = float(solver.core)
    cp_core = -gamma_v**2/(4*np.pi**2*rc**2*u**2)
    wake = surface.wake["real"]["middle"].reshape(-1, n + 1, 3)
    traj = np.concatenate([wake[:, 0, :], wake[:, -1, :]])                # the two edge filaments of the wake
    for key in ("left", "right"):
        w = surface.wake["real"].get(key)
        if w is not None and np.size(w):
            traj = np.concatenate([traj, w])
    depth_min = float(np.abs(traj[:, 2]).min())
    sigma_v = 2*g*depth_min/u**2
    sub_atm = -cp_core >= sigma_v
    path = depth_min <= k_path*rc
    return (sub_atm and path), dict(gamma_v=gamma_v, core_radius=rc, cp_core=cp_core, depth_min=depth_min,
                                    sigma_at_min_depth=sigma_v, sub_atmospheric=bool(sub_atm), path=bool(path),
                                    validated=False)


def assess(surface, solver, data, g=9.81, nu=1.0e-6, band=0.5, k_path=1.0, submerged=False, seal="effective"):
    """
    Full inception assessment of one surface. -> dict
        state   : per-section quantities (section_state)
        prone_fraction : fraction of the immersed span that satisfies condition A
        routes  : {"nose": (open, seed), "tail": (open, details), "tip": (open, details) or None}
        incepts : True when any route is open
        route   : name of the first open route, or None
    `submerged` selects the tip-vortex route (for an element that does not pierce the surface).
    """
    st = section_state(surface, solver, data, g, nu, band, seal)
    nose = nose_route(st)
    tail = tail_route(st, surface, data)
    tip = tip_vortex_route(surface, solver, g, k_path) if submerged else None
    routes = {"nose": nose, "tail": tail, "tip": tip}
    opened = [k for k, v in routes.items() if v is not None and v[0]]
    return dict(state=st, prone_fraction=float(np.mean(st["prone"])), routes=routes,
                incepts=len(opened) > 0, route=(opened[0] if opened else None))


def report(result, data=None):
    """Print an assessment in a form a fluid dynamicist can check line by line."""
    st = result["state"]
    print(f"u = {st['u']:.3f} m/s  h = {st['h']:.3f} m  c = {st['c_mean']:.3f} m  Fn_h = {st['Fn_h']:.2f}  "
          f"Fn_c = {st['Fn_c']:.2f}  AR_h = {st['AR_h']:.2f}")
    if data is not None:
        print(f"section data: {data.name}  [{data.status}]")
    print(f"separation gate: {st['seal']} incidence")
    print(" depth/c   sigma_c   alpha_eff  alpha_sep   Cp_min  separated  sub-atm  prone")
    for i in range(len(st["depth"])):
        print(f" {st['depth'][i]/st['chord'][i]:7.3f} {st['sigma_c'][i]:9.3f} {st['alpha_eff'][i]:10.2f} "
              f"{st['alpha_sep'][i]:10.2f} {st['cp_min'][i]:8.2f} {str(st['separated'][i]):>9s} "
              f"{str(st['sub_atmospheric'][i]):>8s} {str(st['prone'][i]):>6s}")
    print(f"prone fraction of the span: {result['prone_fraction']:.2f}")
    for k, v in result["routes"].items():
        if v is None:
            continue
        print(f"route {k:4s}: {'OPEN' if v[0] else 'closed'}  {v[1]}")
    print("inception:", result["incepts"], "via", result["route"])
