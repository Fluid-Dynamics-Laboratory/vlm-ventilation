"""
Two-dimensional section data for the ventilation inception model.

A SectionData object answers, for one aerofoil section, the questions the inception model asks
of a section at a given effective incidence and Reynolds number:

    is the suction side separated?          -> alpha_sep(Re), separated(alpha, Re)
    how low is the pressure on it?          -> cp_min(alpha, Re)
    how far aft does the flow stay attached -> x_sep(alpha, Re)      (viscous tables only)
    how much to trust the answer            -> confidence(alpha, Re)  (viscous tables only)

and, for a blunt trailing edge, the base pressure coefficient cp_base.

Two table formats (JSON) are read:

  format 1, one-dimensional (src/section_data/naca0009.json, PROVISIONAL)
    alpha_deg, cp_min_inviscid (against alpha), alpha_sep {"Re", "alpha_deg"}, cp_base_blunt_te,
    lift_slope_2d. Inviscid Cp_min from a vortex panel method; separation angles read from
    published lift curves (docs/inception/make_naca0009_table.py).

  format 2, two-dimensional (src/section_data/naca0009_neuralfoil.json)
    alpha_deg, Re_grid, and the arrays cl, cp_min, x_sep, confidence of shape (len(Re_grid),
    len(alpha_deg)); alpha_sep {"Re", "alpha_deg"} derived from the lift curves; provenance.
    Built by `SectionData.from_neuralfoil` from NeuralFoil (P. Sharpe, MIT licence), a pure
    Python surrogate of XFOIL trained on about eight million XFOIL runs, which returns the lift,
    the transition point, the boundary-layer edge velocity and shape factor at 32 stations, and a
    confidence value. Cp_min is 1 - (u_e/u_inf)^2 of the suction side with the boundary layer
    included; alpha_sep is the incidence of the first lift maximum (stall); x_sep is the first
    station downstream of transition where the shape factor exceeds 2.8 (turbulent separation).
    Confidence below 0.5 marks conditions where XFOIL itself would not have converged and the
    values are extrapolated.

Values are interpolated linearly in alpha and in log10(Re) and held constant outside the table.
Any table in either format can be written by XFOIL polars; the format is the exchange interface.
"""
import json, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
H_SEPARATION = 2.8          # shape factor of turbulent separation (XFOIL's criterion is about this value)
CONFIDENCE_LOW = 0.5        # below this the surrogate is extrapolating


class SectionData:
    def __init__(self, table):
        if isinstance(table, str):
            table = json.load(open(table))
        self.name    = table.get("section", "unnamed")
        self.status  = table.get("status", "")
        self.source  = table.get("source", table.get("provenance", ""))
        self._alpha  = np.asarray(table["alpha_deg"], float)
        self._re_sep = np.log10(np.asarray(table["alpha_sep"]["Re"], float))
        self._asep   = np.asarray(table["alpha_sep"]["alpha_deg"], float)
        self.cp_base = float(table.get("cp_base_blunt_te", -0.3))
        self.a0      = float(table.get("lift_slope_2d", 2*np.pi))
        if "Re_grid" in table:                       # format 2
            self._re = np.log10(np.asarray(table["Re_grid"], float))
            self._cpmin = np.asarray(table["cp_min"], float)             # (nRe, nAlpha)
            self._cl = np.asarray(table["cl"], float)
            self._xsep = np.asarray(table.get("x_sep", np.ones_like(self._cpmin)), float)
            self._conf = np.asarray(table.get("confidence", np.ones_like(self._cpmin)), float)
            self.viscous = True
        else:                                        # format 1
            self._re = None
            self._cpmin = np.asarray(table["cp_min_inviscid"], float)
            self._cl = None; self._xsep = None; self._conf = None
            self.viscous = False

    # ------------------------------------------------------------------ constructors
    @classmethod
    def naca0009(cls, source="neuralfoil"):
        """The NACA 0009 table: 'neuralfoil' (viscous, default) or 'provisional' (inviscid Cp, published stall)."""
        f = "naca0009_neuralfoil.json" if source == "neuralfoil" else "naca0009.json"
        path = os.path.join(_HERE, "section_data", f)
        if not os.path.exists(path) and source == "neuralfoil":
            path = os.path.join(_HERE, "section_data", "naca0009.json")
        return cls(path)

    @classmethod
    def from_neuralfoil(cls, coordinates, name, Re=(1e5, 2e5, 5e5, 7e5, 1e6, 1.6e6, 3e6, 6e6, 1e7),
                        alpha_deg=None, model_size="xlarge", n_crit=9.0, cp_base=-0.3, save=None):
        """
        Build a format-2 table from NeuralFoil for the section given by its coordinates (Selig
        order, trailing edge -> upper surface -> leading edge -> lower surface -> trailing edge).
        Re: chord Reynolds numbers of the grid; alpha_deg: incidences (default 0 to 25 by 0.5).
        save: path of the JSON to write. -> SectionData
        """
        import neuralfoil as nf
        alphas = np.arange(0.0, 25.01, 0.5) if alpha_deg is None else np.asarray(alpha_deg, float)
        Re = np.asarray(Re, float)
        xs = np.asarray(nf.bl_x_points, float)
        cl = []; cpmin = []; xsep = []; conf = []; asep = []
        for re in Re:
            r = nf.get_aero_from_coordinates(np.asarray(coordinates, float), alpha=alphas, Re=float(re),
                                             n_crit=n_crit, model_size=model_size)
            ue = np.stack([r[f"upper_bl_ue/vinf_{k}"] for k in range(len(xs))], axis=1)
            H = np.stack([r[f"upper_bl_H_{k}"] for k in range(len(xs))], axis=1)
            xtr = np.asarray(r["Top_Xtr"], float)
            c = np.asarray(r["CL"], float)
            # first lift maximum: stall, the separation incidence of the inception model
            drop = np.where(c[1:] < c[:-1] - 1e-3)[0]
            i_stall = int(drop[0]) if len(drop) else int(np.argmax(c))
            asep.append(float(alphas[i_stall]))
            xs_row = []
            for i in range(len(alphas)):
                m = (H[i] > H_SEPARATION) & (xs > xtr[i])
                xs_row.append(float(xs[np.argmax(m)]) if m.any() else 1.0)
            cl.append(c.tolist()); cpmin.append((1 - (ue**2).max(axis=1)).tolist())
            xsep.append(xs_row); conf.append(np.asarray(r["analysis_confidence"], float).tolist())
        table = {
            "section": name,
            "status": "viscous section data from NeuralFoil; check `confidence` where it falls below 0.5",
            "source": f"NeuralFoil {getattr(nf, '__version__', '')} model {model_size}, n_crit {n_crit}; "
                      "alpha_sep = incidence of the first lift maximum; Cp_min = 1 - max(u_e/u_inf)^2 on the suction side; "
                      f"x_sep = first station downstream of transition with shape factor > {H_SEPARATION}",
            "alpha_deg": alphas.tolist(), "Re_grid": Re.tolist(),
            "cl": cl, "cp_min": cpmin, "x_sep": xsep, "confidence": conf,
            "alpha_sep": {"Re": Re.tolist(), "alpha_deg": asep},
            "cp_base_blunt_te": cp_base, "lift_slope_2d": 2*np.pi,
        }
        if save:
            json.dump(table, open(save, "w"), indent=1)
        return cls(table)

    # ------------------------------------------------------------------ queries
    def _interp2(self, arr, alpha_deg, re):
        a = np.abs(np.asarray(alpha_deg, float)); lre = np.log10(np.asarray(re, float))
        a, lre = np.broadcast_arrays(a, lre)
        out = np.empty(a.shape)
        for idx in np.ndindex(a.shape):
            row = np.array([np.interp(a[idx], self._alpha, arr[k]) for k in range(len(self._re))])
            out[idx] = np.interp(lre[idx], self._re, row)
        return out

    def cp_min(self, alpha_deg, re=1e6):
        """Minimum pressure coefficient at incidence |alpha| [deg] and Reynolds number re. -> array"""
        if self.viscous:
            return self._interp2(self._cpmin, alpha_deg, re)
        return np.interp(np.abs(np.asarray(alpha_deg, float)), self._alpha, self._cpmin)

    def alpha_sep(self, re):
        """Separation (stall) incidence [deg] at Reynolds number re. -> array"""
        return np.interp(np.log10(np.asarray(re, float)), self._re_sep, self._asep)

    def separated(self, alpha_deg, re):
        """True where the suction side is separated: |alpha| >= alpha_sep(Re). -> bool array"""
        return np.abs(np.asarray(alpha_deg, float)) >= self.alpha_sep(re)

    def x_sep(self, alpha_deg, re):
        """Chordwise position of turbulent separation (1 = attached to the trailing edge). -> array"""
        if not self.viscous:
            return np.ones(np.broadcast(np.asarray(alpha_deg), np.asarray(re)).shape)
        return self._interp2(self._xsep, alpha_deg, re)

    def confidence(self, alpha_deg, re):
        """Confidence of the section data at the condition (1 for a table without the field). -> array"""
        if not self.viscous:
            return np.ones(np.broadcast(np.asarray(alpha_deg), np.asarray(re)).shape)
        return self._interp2(self._conf, alpha_deg, re)

    def cl(self, alpha_deg, re):
        """Sectional lift coefficient from the table (viscous tables only). -> array"""
        if not self.viscous:
            return self.a0*np.deg2rad(np.abs(np.asarray(alpha_deg, float)))
        return self._interp2(self._cl, alpha_deg, re)
