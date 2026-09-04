"""
Ventilated cavity on the vortex lattice, and the flow-regime machine.

This module gives the vortex lattice method a cavity: a sheet cavity on the suction side of
a lifting surface, at a prescribed pressure, whose chordwise extent on every spanwise section
is solved for, and whose effect on the circulation, and hence on the loads, is a result of the
solve rather than a prescribed twist. It also carries the hysteretic regime machine (fully
wetted, partially ventilated, fully ventilated) that decides when a cavity forms, persists and
washes out, following Harwood, Young & Ceccio, J. Fluid Mech. 800 (2016).

Formulation (linearised thin-section cavity theory on the lattice)
------------------------------------------------------------------
The wing is the ring-vortex lattice of VLMSolver. A ventilated panel carries in addition a
constant-strength source sheet q, the transpiration that gives the cavity its thickness
(Acosta 1955; Tulin 1953; Kinnas & Fine 1993 for the panel form). Two conditions hold on a
cavity panel instead of one:

    kinematic, on the wetted (pressure) side of the sheet
        (u_inf + v_rings + v_sources) . n_p = 0
      the source sheet emits q/2 on each side, so the pressure-side normal velocity is
      evaluated at a point displaced by -epsilon n_s from the control point and contains the
      own-panel term automatically;

    dynamic, on the cavity (suction) side
        u_inf . tau + v_wake . tau + v_sources . tau + gamma_i / 2 = u_inf sqrt(1 + sigma_c)
      gamma_i = (Gamma_i - Gamma_{i-1}) / dc_i is the bound vorticity per unit chord of the
      panel, whose jump across the sheet is the only tangential velocity a planar lattice
      induces on itself; sigma_c is the cavitation number at the panel depth,
        sigma_c(z) = 2 g |z| / u_inf^2 + dsigma   (dsigma = 0 for natural ventilation).

Wetted panels keep the kinematic condition only. The unknowns are the ring strengths of all
panels and the source strengths of the cavity panels; the system is square.

Cavity extent. The length L_j of the cavity on section j is taken from the sectional theory:
Psi_j = sigma_c(z_j) / (2 alpha_eff,j) with alpha_eff,j = Cl_j/(2 pi) of the wetted solution
of the same step, and L_j = L(Psi_j) from the blend of Acosta's partial-cavity and Tulin's
supercavity solutions of vent_section.cavity_length (Harwood et al. 2016). The cavity covers
the whole panels up to L_j and the load is interpolated between the two bracketing panel
counts, so it is continuous in the incidence. The thickness, h(x) = (1/u_inf) integral q dx,
and its value at the cavity end (the closure residual) are diagnostics. A closure iteration
on the lattice (the length adjusted until the thickness vanishes at the cavity end) was tried
and not retained: on a collocated source-vortex lattice the source strengths alternate in sign
from panel to panel and the thickness at closure is not a usable residual (docs/cavity/
README.md, Sect. 4).

Two branches per section, by the range of validity of the linear theory:
  L_j <= L_MAX (0.5)  partial cavity: the mixed lattice solve above; the lift follows from
                       the redistributed circulation (verified against Acosta, Sect. 3 of the
                       README, within a few per cent);
  L_j >  L_MAX         long or supercavity: the linear closed-cavity model truncated at the
                       trailing edge is not valid, so the sectional lift slope a0(L_j) of
                       vent_section.lift_slope is imposed on the section by scaling the
                       free-stream term of its kinematic condition by a0/(2 pi), which is the
                       effective-incidence correction of the internship report in a
                       conservative form. The two branches agree to about 5 per cent at L_MAX.

Loads. The lift follows from the circulation by the linearised Kutta-Joukowski theorem of the
base solver. The centre of pressure of each section is computed from the chordwise
distribution of bound vorticity on the partial-cavity branch and from
vent_section.centre_of_pressure on the long-cavity branch.

Regime machine (Harwood et al. 2016; the design of Viola's bem-fem-fsi vent_cavity.py)
---------------------------------------------------------------------------------------
Per surface: regime in {FW, PV, FV}; cavity length L_j per section, continuous; a growth
rate limit (chords of cavity per chord of travel) so that a cavity forms over a few
convective times rather than in one step. Formation needs an inception signal, which is a
callable (surface, solver) -> assessment with keys `incepts` and `state["prone"]` (the
interface of VLMInception.assess), or the `force` method. Persistence needs only that the
wetted suction at the leading-edge panel exceeds sigma_c (a cavity can exist); it never asks
again for the seal, and that asymmetry is the hysteresis. Elimination follows the re-entrant
jet as in Sect. 4 of Harwood et al.: the local angle of the closure line from the flow at the
mid-depth section, from the equilibrium lengths L_j(z) with vent_section.closure_angle, exceeds
45 degrees. Above the formation boundary an unstable cavity persists and sheds (partially
ventilated); below it, it washes out at the rate limit. The regime label is vent_section.regime(D, phi_bar, h) with D the extent of the
ventilated sections.

The sectional module vent_section.py (Acosta, Tulin, the fits of Harwood et al., the
closure angle, the washout Froude number) is the closed-form reference the lattice cavity is
verified against (docs/cavity/verify_acosta.py) and supplies the first guess of the extent.

Usage
-----
    solver = CavitySolver([surface], u_inf, boundary, ratio, a_ratio, g=9.81)
    solver.vent.force(surface, strips=None)          # or: solver.vent.inception = callable
    solver._time_sim(t, dt, "classic"); solver._kuttas_loads()
    solver.vent.report(surface)
"""
import numpy as np

import vent_section as vs
from VLMSolver import VLMSolver

EPS_SIDE = 1.0e-6        # displacement of the evaluation points off the sheet, in chords
CLOSURE_TOL = 1.0e-3     # closure residual, thickness at the cavity end over chord, considered zero
MAX_SWEEPS = 40          # sweeps of the extent iteration over all sections (extent="closure" only)
L_MAX = 0.5              # longest cavity solved on the lattice (Acosta's limit); beyond, the sectional lift slope is imposed
L_LOAD_CAP = 10.0        # cavity length beyond which the sectional lift slope is taken at this value (within 1 per cent of pi/2)


# ---------------------------------------------------------------- source panel kernel

def _frame(corners, normal):
    """Local frame of a planar quadrilateral: origin, chordwise ex (LE -> TE), ey, ez = normal."""
    c = np.asarray(corners, float)
    o = c.mean(axis=0)
    ez = np.asarray(normal, float); ez = ez/np.linalg.norm(ez)
    ex = 0.5*((c[2] + c[3]) - (c[0] + c[1]))
    ex = ex - (ex @ ez)*ez; ex = ex/np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return o, ex, ey, ez


def source_panel_velocity(corners, normal, points):
    """
    Velocity induced at `points` (P,3) by a constant-strength source quadrilateral of unit
    strength (Hess & Smith 1962; Katz & Plotkin, Sect. 10.4.1). The in-plane components use
    the edge logarithms; the normal component is the solid angle of the panel at the point,
    w = Omega/(4 pi), which is robust for edges parallel to either in-plane axis. -> (P,3)
    """
    o, ex, ey, ez = _frame(corners, normal)
    c = np.asarray(corners, float) - o
    xk = c @ ex; yk = c @ ey
    d = np.asarray(points, float) - o
    x = d @ ex; y = d @ ey; z = d @ ez
    u = np.zeros(len(points)); v = np.zeros(len(points))
    for k in range(4):
        k1 = (k + 1) % 4
        dk = np.hypot(xk[k1] - xk[k], yk[k1] - yk[k])
        if dk < 1e-14:
            continue
        rk = np.sqrt((x - xk[k])**2 + (y - yk[k])**2 + z**2)
        rk1 = np.sqrt((x - xk[k1])**2 + (y - yk[k1])**2 + z**2)
        num = np.maximum(rk + rk1 - dk, 1e-300); den = rk + rk1 + dk
        lg = np.log(num/den)
        u += (yk[k1] - yk[k])/dk*lg
        v += (xk[k] - xk[k1])/dk*lg
    # solid angle by the fan of two triangles (Van Oosterom & Strackee 1983)
    P = np.asarray(points, float)
    omega = np.zeros(len(points))
    for tri in ((0, 1, 2), (0, 2, 3)):
        a = np.asarray(corners, float)[tri[0]] - P
        b = np.asarray(corners, float)[tri[1]] - P
        cc = np.asarray(corners, float)[tri[2]] - P
        la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(cc, axis=1)
        numer = np.einsum("ij,ij->i", a, np.cross(b, cc))
        denom = la*lb*lc + np.einsum("ij,ij->i", a, b)*lc + np.einsum("ij,ij->i", a, cc)*lb + np.einsum("ij,ij->i", b, cc)*la
        omega += 2*np.arctan2(numer, denom)
    w = omega/(4*np.pi)
    inv4pi = 1/(4*np.pi)
    return (u*inv4pi)[:, None]*ex + (v*inv4pi)[:, None]*ey + w[:, None]*ez


# ---------------------------------------------------------------- inception gates and state transfer

def stall_gate(alpha_stall_deg):
    """
    The simplest inception signal: the surface incepts on all its sections when its geometric incidence
    (angle of attack plus drift) reaches the stall angle, the incidence-only boundary of Harwood et al.
    (2016). The full per-section model of branch vent/inception has the same call signature and replaces
    this at merge. -> callable(surface, solver) -> dict(incepts, route, state={'prone'})
    """
    def gate(surface, solver):
        alpha = np.degrees(abs(surface.aoa) + abs(surface.drift))
        inc = alpha >= alpha_stall_deg
        return dict(incepts=bool(inc), route="stall" if inc else None, state=dict(prone=np.full(surface.N, inc)))
    return gate


def carry_state(old_solver, old_surface, new_solver, new_surface):
    """Copy the ventilation state of a surface to a new solver of the same discretisation (parameter sweeps)."""
    a = old_solver.vent.state[id(old_surface)]; b = new_solver.vent.state[id(new_surface)]
    for k in ("regime", "route", "unstable", "phi_bar"):
        b[k] = a[k]
    for k in ("L", "active", "m_c", "open"):
        b[k] = np.array(a[k], copy=True)
    b["forced"] = None if a["forced"] is None else np.array(a["forced"], copy=True)


# ---------------------------------------------------------------- the ventilation state

class Ventilation:
    """
    Regime state of every surface of a CavitySolver. Built once by the solver; advanced by the
    solver at the end of each step. Per surface (keyed by id):
        regime   : 'FW' | 'PV' | 'FV'
        L        : (N,) cavity length over chord of each section (0 = wetted)
        m_c      : (N,) whole cavity panels of each section
        open     : (N,) bool, cavity reaching the trailing edge with positive thickness
        active   : (N,) bool, section ventilated
        phi_bar  : local closure angle at the mid-depth section [rad] or None (the stability measure);
                   phi_mean the least-squares mean angle of the whole closure line
        route    : how the cavity formed
        history  : list of (step, regime, D/h, phi_bar, CL) tuples
    """

    def __init__(self, solver, g=9.81, dsigma=0.0, rate=0.2, phi_crit=vs.PHI_CRIT, inception=None, washout=True):
        self.solver = solver
        self.g = g; self.dsigma = dsigma; self.rate = rate; self.phi_crit = phi_crit
        self.washout = washout              # False disables the re-entrant-jet elimination (two-dimensional verification)
        self.washout_time = 3.0             # chords of travel an unstable cavity survives before the flow rewets
        self.inception = inception          # callable(surface, solver) -> assessment, or None
        self.state = {}
        for surface in solver.surfaces:
            n = surface.N
            self.state[id(surface)] = dict(regime="FW", L=np.zeros(n), m_c=np.zeros(n, int), open=np.zeros(n, bool),
                                           active=np.zeros(n, bool), phi_bar=None, route=None, history=[],
                                           forced=None, unstable=False, q=None, closure=np.zeros(n), xcp=np.full(n, np.nan))

    def force(self, surface, strips=None):
        """Force inception on the given sections (all when None), the perturbation route of the experiments."""
        st = self.state[id(surface)]
        st["forced"] = np.ones(surface.N, bool) if strips is None else np.asarray(strips, bool)
        st["route"] = "forced"

    def set_regime(self, surface, regime):
        """Impose a branch of the bi-stable flow: 'FW' clears the cavity, 'FV' forces all sections."""
        if regime == "FW":
            st = self.state[id(surface)]
            st.update(L=np.zeros(surface.N), m_c=np.zeros(surface.N, int), active=np.zeros(surface.N, bool),
                      open=np.zeros(surface.N, bool), regime="FW", forced=None, unstable=False)
        else:
            self.force(surface, None)

    def report(self, surface):
        st = self.state[id(surface)]
        print(f"regime {st['regime']}  route {st['route']}  phi(mid-depth) = "
              f"{np.degrees(st['phi_bar']) if st['phi_bar'] is not None else float('nan'):.1f} deg  unstable {st['unstable']}")
        print(" section  depth/c    sigma_c    L/c   open  closure_res  e_cp")
        geo = self.solver._cavity_geo[id(surface)]
        for j in range(surface.N):
            print(f" {j:6d} {geo['depth_sec'][j]/geo['chord'][j]:9.3f} {geo['sigma_sec'][j]:10.4f} {st['L'][j]:6.3f} "
                  f"{str(bool(st['open'][j])):>5s} {st['closure'][j]:12.4f} {st['xcp'][j]:6.3f}")


# ---------------------------------------------------------------- the solver

class CavitySolver(VLMSolver):
    """
    VLMSolver with a ventilated cavity on the lattice and the regime machine. Same constructor
    plus g (gravity), dsigma (cavity pressure excess over atmospheric, in units of the dynamic
    pressure; 0 for natural ventilation), rate (cavity growth, chords per chord of travel) and
    inception (callable or None).
    """

    def __init__(self, surfaces, u_inf, boundary, ratio, a_ratio, g=9.81, dsigma=0.0, rate=0.2, inception=None, washout=True):
        super().__init__(surfaces, u_inf, boundary, ratio, a_ratio)
        self.vent = Ventilation(self, g, dsigma, rate, inception=inception, washout=washout)
        self._cavity_geo = {}
        self._source_influence = None
        self._prepare_geometry()

    # ------------------------------------------------ geometry and influence of the source sheet
    def _prepare_geometry(self):
        u = float(np.linalg.norm(self.U)); uhat = self.U/u
        for surface in self.surfaces:
            m, n = surface.M, surface.N
            panels = surface.wing_panels["real"]
            ctr = np.array([p.ctr for p in panels]); nrm = np.array([p.normal for p in panels])
            tau = np.array([p.chord/np.linalg.norm(p.chord) for p in panels])
            dc = np.array([np.linalg.norm(p.chord) for p in panels])
            chord = (m*dc).reshape(m, n)[0]
            s_side = np.sign(nrm @ uhat); s_side[s_side == 0] = 1.0        # +1: normal points to the suction side
            depth = np.abs(ctr[:, 2])
            sigma = 2*self.vent.g*depth/u**2 + self.vent.dsigma
            # chordwise position of the panel centre, in chords from the leading edge, per section
            xc = ((np.arange(m) + 0.5)/m)[:, None]*np.ones((1, n))
            # coordinate along the span of each section: the depth for a vertical strut (plan 0), the spanwise
            # position from one tip for a horizontal element (plan 1); used for the closure line and the cavity extent
            span_sec = depth.reshape(m, n).mean(axis=0) if surface.plan == 0 else ctr[:, 1].reshape(m, n).mean(axis=0)
            span_sec = span_sec - span_sec.min()
            ds = (span_sec.max() - span_sec.min())/max(n - 1, 1)
            self._cavity_geo[id(surface)] = dict(ctr=ctr, normal=nrm, tau=tau, dc=dc, chord=chord, s_side=s_side,
                                                 depth=depth, sigma=sigma, depth_sec=depth.reshape(m, n).mean(axis=0),
                                                 sigma_sec=sigma.reshape(m, n).mean(axis=0), xc=xc.reshape(-1),
                                                 span_sec=span_sec, span_total=span_sec.max() + ds, ds=ds)
        # source influence of every panel at the pressure-side (normal) and suction-side (tangential) points
        offs = 0
        K = self.N_tot
        Sn = np.zeros((K, K)); St = np.zeros((K, K))
        allctr = self.controls; allnrm = self.normals
        for surface in self.surfaces:
            g = self._cavity_geo[id(surface)]
            m, n = surface.M, surface.N
            eps = EPS_SIDE*g["chord"].mean()
            # evaluation points of THIS surface, displaced to its pressure and suction sides
            p_press = g["ctr"] - (g["s_side"]*eps)[:, None]*g["normal"]
            p_suct = g["ctr"] + (g["s_side"]*eps)[:, None]*g["normal"]
            cols = slice(offs, offs + m*n)
            # rows: all control points of all surfaces; other surfaces are far, use their plain control points
            pts_n = allctr.copy(); pts_n[cols] = p_press
            pts_t = allctr.copy(); pts_t[cols] = p_suct
            for jj, p in enumerate(surface.wing_panels["real"]):
                vn = source_panel_velocity(p.pnt, p.normal, pts_n)
                vt = source_panel_velocity(p.pnt, p.normal, pts_t)
                Sn[:, offs + jj] = np.einsum("ij,ij->i", vn, allnrm)
                St[cols, offs + jj] = np.einsum("ij,ij->i", vt[cols], g["tau"])
            offs += m*n
        self._Sn, self._St = Sn, St

    # ------------------------------------------------ the mixed solve
    def _mixed_solve(self, m_c_all, slope_ratio=None):
        """
        Solve rings + sources for the cavity extents m_c_all (dict id(surface) -> (N,) ints).
        slope_ratio: optional dict id(surface) -> (N,) factors a0(L)/(2 pi) applied to the free-stream term of the
        kinematic condition of each section (long-cavity branch; 1 leaves the section untouched).
        -> Gamma (N_tot,), q (N_tot,) with zeros on wetted panels, closure residual per section (dict), open flags (dict)
        """
        A = self.A; K = self.N_tot
        b = self.b.copy()
        if slope_ratio is not None:
            offs = 0
            for surface in self.surfaces:
                m, n = surface.M, surface.N
                r = np.tile(np.asarray(slope_ratio[id(surface)], float), m)
                un = self.normals[offs:offs + m*n] @ self.U
                b[offs:offs + m*n] += (1.0 - r)*un            # b = -(v_wake + r u_inf).n
                offs += m*n
        u = float(np.linalg.norm(self.U))
        cav = []; dyn_rows = []; rhs_dyn = []
        offs = 0
        v_wake_t = self._wake_tangential()
        for surface in self.surfaces:
            g = self._cavity_geo[id(surface)]; m, n = surface.M, surface.N
            m_c = m_c_all[id(surface)]
            sgn = self._loading_sign(surface)
            for j in range(n):
                for i in range(int(m_c[j])):
                    k = i*n + j; kk = offs + k
                    cav.append(kk)
                    row = np.zeros(K)
                    row[kk] += sgn[j]/(2*g["dc"][k])
                    if i > 0:
                        row[offs + (i - 1)*n + j] -= sgn[j]/(2*g["dc"][k])
                    dyn_rows.append(row)
                    rhs_dyn.append(u*(1 + 0.5*g["sigma"][k]) - (self.U @ g["tau"][k]) - v_wake_t[kk])   # linearised: u sqrt(1+sigma) ~ u (1 + sigma/2), as in Acosta (1955)
            offs += m*n
        if not cav:
            Gamma = self._inv_A @ b
            return Gamma, np.zeros(K), {id(s): np.zeros(s.N) for s in self.surfaces}, {id(s): np.zeros(s.N, bool) for s in self.surfaces}
        cav = np.array(cav); nc = len(cav)
        M = np.zeros((K + nc, K + nc)); r = np.zeros(K + nc)
        M[:K, :K] = A
        M[:K, K:] = self._Sn[:, cav]
        M[K:, :K] = np.array(dyn_rows)
        M[K:, K:] = self._St[cav][:, cav]
        r[:K] = b; r[K:] = rhs_dyn
        x = np.linalg.solve(M, r)
        Gamma = x[:K]; q = np.zeros(K); q[cav] = x[K:]
        closure = {}; opened = {}
        offs = 0
        for surface in self.surfaces:
            g = self._cavity_geo[id(surface)]; m, n = surface.M, surface.N
            m_c = m_c_all[id(surface)]
            res = np.zeros(n); op = np.zeros(n, bool)
            for j in range(n):
                if m_c[j] > 0:
                    ks = offs + np.arange(int(m_c[j]))*n + j
                    res[j] = float(np.sum(q[ks]*g["dc"][ks - offs])/(u*g["chord"][j]))      # h(end)/c
                    op[j] = (m_c[j] >= m) and res[j] > CLOSURE_TOL
            closure[id(surface)] = res; opened[id(surface)] = op
            offs += m*n
        return Gamma, q, closure, opened

    def _wake_tangential(self):
        """Chordwise velocity induced by the older wake rings at the suction-side points of all panels. -> (N_tot,)"""
        if all(np.size(s.gamma_wake["middle"]) == 0 for s in self.surfaces):
            return np.zeros(self.N_tot)
        c1, c2, gamma_v = self._full_vectorize(False, skip_shed=True)     # the rings shed at this step are in A
        out = np.zeros(self.N_tot); offs = 0
        for surface in self.surfaces:
            g = self._cavity_geo[id(surface)]; m, n = surface.M, surface.N
            v = self._induced_velocity(g["ctr"], c1, c2, gamma_v, self.rc*self.a_ratio, 0.0)
            out[offs:offs + m*n] = np.einsum("ij,ij->i", v, g["tau"])
            offs += m*n
        return out

    def _loading_sign(self, surface):
        """Sign of the sectional circulation of the wetted baseline, so that gamma_i/2 adds speed on the suction side. -> (N,)"""
        base = getattr(self, "_gamma_wetted", None)
        if base is None:
            return np.ones(surface.N)
        offs = self._offset(surface); m, n = surface.M, surface.N
        gte = base[offs:offs + m*n].reshape(m, n)[-1]
        sgn = np.sign(gte); sgn[sgn == 0] = 1.0
        # the suction side is by definition the faster side, so the jump +|gamma_i|/2 is added whatever the
        # orientation of the panel normal; the sign only converts Gamma_i - Gamma_{i-1} to a magnitude
        return sgn

    def _offset(self, surface):
        offs = 0
        for s in self.surfaces:
            if s is surface:
                return offs
            offs += s.M*s.N
        raise ValueError("surface not in solver")

    # ------------------------------------------------ sectional extent and the two branches
    def _sectional_targets(self, surface):
        """Cavity length L(Psi) of every section from the wetted baseline of this step, and the branch of each. -> (L, alpha_eff)"""
        g = self._cavity_geo[id(surface)]; m, n = surface.M, surface.N
        offs = self._offset(surface)
        u = float(np.linalg.norm(self.U))
        gte = np.abs(self._gamma_wetted[offs:offs + m*n].reshape(m, n)[-1])
        alpha_eff = np.maximum(gte/(np.pi*u*g["chord"]), 1e-9)              # Cl = 2 Gamma/(u c) = 2 pi alpha_eff
        psi = np.maximum(g["sigma_sec"], 0.0)/(2*alpha_eff)
        L = np.where(psi > 0, vs.cavity_length(np.maximum(psi, 1e-12)), vs.L_FIT_MAX)
        self.vent.state[id(surface)]["L_stability"] = 2.31/np.maximum(psi, 2.31/vs.L_FIT_MAX)   # long-cavity relation (1.8)
        return np.asarray(L, float), alpha_eff

    def _solve_with_lengths(self, lengths):
        """
        Solve the lattice for given cavity lengths per section (dict id -> (N,), 0 = wetted).
        Sections with 0 < L <= L_MAX get the mixed solve with the load interpolated between the bracketing panel
        counts; sections with L > L_MAX get the sectional slope a0(L)/(2 pi) on their free-stream term.
        -> Gamma, q, info dict per surface: L, m_c, open, closure, branch ('wet'|'lattice'|'sectional')
        """
        m_lo = {}; m_hi = {}; frac = {}; ratio = {}; branch = {}
        need_hi = False
        for surface in self.surfaces:
            m, n = surface.M, surface.N
            L = np.asarray(lengths[id(surface)], float)
            lat = (L > 0) & (L <= L_MAX)
            sec = L > L_MAX
            x = np.clip(L*m, 0, m)
            lo = np.floor(x).astype(int); hi = np.minimum(lo + 1, m)
            f = x - lo
            lo = np.where(lat, lo, 0); hi = np.where(lat, hi, 0); f = np.where(lat, f, 0.0)
            m_lo[id(surface)] = lo; m_hi[id(surface)] = hi; frac[id(surface)] = f
            r = np.ones(n); r[sec] = vs.lift_slope(L[sec])/(2*np.pi)
            ratio[id(surface)] = r
            branch[id(surface)] = np.where(sec, "sectional", np.where(lat, "lattice", "wet"))
            need_hi = need_hi or bool(np.any(lat & (f > 1e-9)))
        G0, q0, c0, o0 = self._mixed_solve(m_lo, ratio)
        if need_hi:
            G1, q1, c1, o1 = self._mixed_solve(m_hi, ratio)
            w = np.zeros(self.N_tot); offs = 0
            for surface in self.surfaces:
                m, n = surface.M, surface.N
                w[offs:offs + m*n] = np.tile(frac[id(surface)], m); offs += m*n
            Gamma = (1 - w)*G0 + w*G1; q = (1 - w)*q0 + w*q1
        else:
            Gamma, q = G0, q0
        info = {}
        for surface in self.surfaces:
            f = frac[id(surface)]
            clo = c0[id(surface)] if not need_hi else (1 - f)*c0[id(surface)] + f*c1[id(surface)]
            info[id(surface)] = dict(L=np.asarray(lengths[id(surface)], float), m_c=m_lo[id(surface)], open=np.asarray(lengths[id(surface)]) > L_MAX,
                                     closure=clo, branch=branch[id(surface)], ratio=ratio[id(surface)])
        return Gamma, q, info

    def _centre_of_pressure(self, surface, Gamma):
        """Centre of pressure of each section forward of mid-chord, in chords, from the chordwise bound vorticity. -> (N,)"""
        g = self._cavity_geo[id(surface)]
        offs = self._offset(surface); m, n = surface.M, surface.N
        gam = Gamma[offs:offs + m*n].reshape(m, n)
        dg = np.diff(np.vstack([np.zeros((1, n)), gam]), axis=0)              # bound vorticity per panel
        xc = g["xc"].reshape(m, n)
        tot = dg.sum(axis=0)
        xcp = np.where(np.abs(tot) > 1e-14, (xc*dg).sum(axis=0)/np.where(np.abs(tot) > 1e-14, tot, 1.0), np.nan)
        return 0.5 - xcp

    # ------------------------------------------------ hooks
    def _solve_step(self, s, dt):
        self._gamma_wetted = self._inv_A @ self.b
        u = float(np.linalg.norm(self.U))
        any_active = any(bool(np.any(self.vent.state[id(sf)]["active"])) for sf in self.surfaces)
        if not any_active:
            self._q = np.zeros(self.N_tot)
            for surface in self.surfaces:
                st = self.vent.state[id(surface)]
                st["L_target"], st["alpha_eff"] = self._sectional_targets(surface)
                st["xcp"] = self._centre_of_pressure(surface, self._gamma_wetted)
                st["branch"] = np.full(surface.N, "wet")
            return self._gamma_wetted
        lengths = {}
        for surface in self.surfaces:
            st = self.vent.state[id(surface)]; g = self._cavity_geo[id(surface)]
            Lt, aeff = self._sectional_targets(surface)
            st["L_target"] = Lt; st["alpha_eff"] = aeff
            dL = self.vent.rate*u*dt/g["chord"]                               # growth limit per step
            L = np.minimum(np.minimum(Lt, L_LOAD_CAP), st["L"] + dL)         # a0(L) is within 1 per cent of pi/2 beyond L_LOAD_CAP
            lengths[id(surface)] = np.where(st["active"], L, 0.0)
        Gamma, q, info = self._solve_with_lengths(lengths)
        for surface in self.surfaces:
            st = self.vent.state[id(surface)]; g = self._cavity_geo[id(surface)]
            inf = info[id(surface)]
            st["L"] = inf["L"]; st["m_c"] = inf["m_c"]; st["open"] = inf["open"]; st["closure"] = inf["closure"]
            st["branch"] = inf["branch"]
            e = self._centre_of_pressure(surface, Gamma)                      # forward of mid-chord, as vent_section
            st["xcp"] = np.where(inf["branch"] == "sectional", vs.centre_of_pressure(inf["L"]), e)
        self._q = q
        return Gamma

    def _after_step(self, s, dt):
        u = float(np.linalg.norm(self.U))
        for surface in self.surfaces:
            st = self.vent.state[id(surface)]; g = self._cavity_geo[id(surface)]
            m, n = surface.M, surface.N
            # can a cavity exist on each section: the sectional length resolves at least one panel
            can_hold = np.asarray(st["L_target"], float) >= 1.0/m
            if not np.any(st["active"]):
                # formation: inception signal or forced
                seed = None
                if st["forced"] is not None:
                    seed = st["forced"] & can_hold
                elif self.vent.inception is not None:
                    self._kuttas_loads()
                    a = self.vent.inception(surface, self)
                    if a["incepts"]:
                        seed = a["state"]["prone"] & can_hold
                        st["route"] = a["route"]
                if seed is not None and np.any(seed):
                    st["active"] = seed.copy(); st["L"] = np.zeros(n); st["unstable"] = False
            else:
                # persistence: a section keeps its cavity while a cavity can exist there
                st["active"] &= can_hold
                st["L"] = np.where(st["active"], st["L"], 0.0)
                # elimination: the re-entrant jet. Harwood et al. (2016, Sect. 4) decide the stability of the
                # fully ventilated cavity at the representative mid-depth section kappa = 0.5, where the jet
                # threatening the cavity is reflected about its deeply submerged part: the cavity washes out when
                # the local slope of the closure line |dx_closure/dz| falls below one, i.e. the local closure
                # angle from the flow exceeds 45 degrees. The slope is that of the equilibrium (target) cavity,
                # uncapped, so a cavity extending beyond the trailing edge keeps its taper.
                act = st["active"]
                if act.sum() >= 3:
                    # the closure line is evaluated with the long-cavity relation L = 2.31/Psi, the one Harwood
                    # et al. use in Sect. 4 to derive the washout boundary; the blend used for the loads flattens
                    # for 1 < L < 3 and would steepen the line by some 25 degrees (docs/cavity/README.md, Sect. 5)
                    span = g["span_sec"][act]
                    xcl = np.asarray(st["L_stability"], float)[act]*g["chord"][act]
                    phi_bar, phi_local = vs.closure_angle(span, xcl)
                    k = int(np.argmin(np.abs(span - 0.5*g["span_total"])))
                    st["phi_bar"] = float(phi_local[k]); st["phi_mean"] = phi_bar
                    st["unstable"] = bool(vs.unstable_closure(st["phi_bar"], self.vent.phi_crit)) and self.vent.washout
                if st["unstable"]:
                    # above the formation boundary the flow stays ventilated and sheds (partially ventilated regime);
                    # below it the cavity washes out at the rate limit and the flow rewets
                    gate_open = st["forced"] is not None
                    if not gate_open and self.vent.inception is not None:
                        gate_open = bool(self.vent.inception(surface, self)["incepts"])
                    if not gate_open:
                        # the re-entrant jet destroys the cavity within a few convective times (Harwood et al.
                        # 2016): the flow rewets once the cavity has been unstable for `washout_time` chords of travel
                        st["unstable_time"] = st.get("unstable_time", 0.0) + u*dt/float(g["chord"].mean())
                        if st["unstable_time"] >= self.vent.washout_time:
                            st["active"][:] = False; st["L"][:] = 0.0
                            st["forced"] = None; st["route"] = "washout"; st["unstable_time"] = 0.0
                    else:
                        st["unstable_time"] = 0.0
                else:
                    st["unstable_time"] = 0.0
            # regime label
            act = st["active"]
            if np.any(act):
                D = float(g["span_sec"][act].max() + g["ds"])              # extent of the ventilated sections, to the far edge of the last one
                h = float(g["span_total"])
                phi = st["phi_bar"] if (st["phi_bar"] is not None and self.vent.washout) else 0.0
                st["regime"] = vs.regime(D, phi, h) if h > 0 else "PV"
            else:
                st["regime"] = "FW"; st["phi_bar"] = None
            st["history"].append((s, st["regime"], float(np.mean(st["active"])), st["phi_bar"]))
