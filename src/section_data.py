"""
Two-dimensional section data for the ventilation inception model.

A SectionData object answers, for one aerofoil section, the two questions the inception
model asks of a section at a given effective incidence and Reynolds number:

    is the suction side separated?          -> alpha_sep(Re), separated(alpha, Re)
    how low is the pressure on it?          -> cp_min(alpha)

and, for a blunt trailing edge, the base pressure coefficient cp_base.

Table format (JSON), see src/section_data/naca0009.json:
    alpha_deg        : list, incidence grid [deg]
    cp_min_inviscid  : list, minimum pressure coefficient at each incidence (inviscid or viscous)
    alpha_sep        : {"Re": list, "alpha_deg": list}, incidence of separation against Reynolds number
    cp_base_blunt_te : float, base pressure coefficient behind a blunt trailing edge
    lift_slope_2d    : float, sectional lift slope [1/rad]
Values are interpolated linearly in alpha and in log10(Re) and held constant outside the table.

The supplied NACA 0009 table is PROVISIONAL: its Cp_min is inviscid and its separation angles
are read from published lift curves (docs/inception/make_naca0009_table.py documents the
sources). Replace it by XFOIL polars in the same format when they are available.
"""
import json, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


class SectionData:
    def __init__(self, table):
        if isinstance(table, str):
            table = json.load(open(table))
        self.name    = table.get("section", "unnamed")
        self.status  = table.get("status", "")
        self._alpha  = np.asarray(table["alpha_deg"], float)
        self._cpmin  = np.asarray(table["cp_min_inviscid"], float)
        self._re     = np.log10(np.asarray(table["alpha_sep"]["Re"], float))
        self._asep   = np.asarray(table["alpha_sep"]["alpha_deg"], float)
        self.cp_base = float(table.get("cp_base_blunt_te", -0.3))
        self.a0      = float(table.get("lift_slope_2d", 2*np.pi))

    @classmethod
    def naca0009(cls):
        return cls(os.path.join(_HERE, "section_data", "naca0009.json"))

    def cp_min(self, alpha_deg):
        """Minimum pressure coefficient at incidence |alpha| [deg]. -> array"""
        return np.interp(np.abs(np.asarray(alpha_deg, float)), self._alpha, self._cpmin)

    def alpha_sep(self, re):
        """Separation incidence [deg] at Reynolds number re (chord based). -> array"""
        return np.interp(np.log10(np.asarray(re, float)), self._re, self._asep)

    def separated(self, alpha_deg, re):
        """True where the suction side is separated: |alpha| >= alpha_sep(Re). -> bool array"""
        return np.abs(np.asarray(alpha_deg, float)) >= self.alpha_sep(re)
