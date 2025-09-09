
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

MU0 = 4e-7*np.pi

@dataclass
class CoreGeom:
    name: str
    Ae_m2: float
    le_m: float
    Ve_m3: float

@dataclass
class Material:
    name: str
    B_limit_T: float

def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    dx = x[1:] - x[:-1]
    avg = 0.5*(y[1:] + y[:-1])
    out[1:] = np.cumsum(avg*dx)
    return out

def build_voltage_knots(rows: List[Tuple[float, float]], t_trans: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(rows) == 0:
        return np.array([0.0, 1e-6]), np.array([0.0, 0.0])
    t = [0.0]
    v = [rows[0][0]]
    V_prev, dt_prev = rows[0]
    t.append(t[-1] + max(0.0, dt_prev)); v.append(V_prev)
    for k in range(1, len(rows)):
        V_k, dt_k = rows[k]
        if (t_trans > 0.0) and (abs(V_k - V_prev) > 1e-12):
            t.append(t[-1] + t_trans); v.append(V_k)
        else:
            if abs(V_k - v[-1]) > 1e-12:
                t.append(t[-1]); v.append(V_k)
        t.append(t[-1] + max(0.0, dt_k)); v.append(V_k)
        V_prev = V_k
    return np.array(t, dtype=float), np.array(v, dtype=float)

def area_volt_seconds(t: np.ndarray, v: np.ndarray) -> float:
    return float(np.trapz(v, t))

def enforce_volt_second_balance(rows: List[Tuple[float, float]], t_trans: float) -> List[Tuple[float, float]]:
    t, v = build_voltage_knots(rows, t_trans)
    A = area_volt_seconds(t, v)
    if abs(A) < 1e-15 or len(rows) == 0:
        return rows
    V_last, dt_last = rows[-1]
    if abs(V_last) < 1e-12:
        return rows
    dt_new = dt_last - A / V_last
    if dt_new > 0:
        new_rows = rows.copy()
        new_rows[-1] = (V_last, dt_new)
        return new_rows
    return rows

def current_from_v_knots(L: float, t: np.ndarray, v: np.ndarray, i0: float = 0.0, periodic: bool = False):
    T = t[-1] if t[-1] > 0 else 1.0
    N = max(5000, int(T/20e-9))
    td = np.linspace(0.0, T, N, endpoint=False)
    vd = np.interp(td, t, v)
    i_rel = _cumtrapz(vd, td)/L
    itot = i0 + i_rel
    if periodic and abs(np.trapz(vd, td)) < 1e-12 and td[-1] > 0:
        drift = itot[-1] - i0
        itot = itot - (drift/td[-1])*td
    return td, itot

def analyze_current_dense(td: np.ndarray, i: np.ndarray) -> Dict[str, float]:
    T = td[-1] if td[-1] > 0 else 1.0
    f = 1.0/T
    Irms = float(np.sqrt(np.trapz(i**2, td)/T))
    Ipk = float(np.max(np.abs(i)))
    Ipp = float(np.max(i) - np.min(i))
    return {"T_s": float(T), "f_Hz": float(f), "Irms_A": Irms, "Ipk_A": Ipk, "Ipp_A": Ipp}

def B_from_v_knots(core: CoreGeom, N: int, t: np.ndarray, v: np.ndarray, center_zero: bool=True):
    Ae = core.Ae_m2
    td = t; vd = v
    Bd = _cumtrapz(vd, td)/(N*Ae)
    if center_zero:
        Bd = Bd - 0.5*(Bd[-1] + Bd[0])
    return td, Bd

def core_library() -> Dict[str, CoreGeom]:
    etd49 = CoreGeom("ETD49", Ae_m2=2.11e-4, le_m=0.114, Ve_m3=2.11e-4*0.114)
    cacc630 = CoreGeom("CACC-630 (nanocrystalline C-core)", Ae_m2=15.58e-4, le_m=32.3e-2, Ve_m3=15.58e-4*32.3e-2)
    return {"ETD49": etd49, "CACC-630 (nanocrystalline C-core)": cacc630}

def material_library() -> Dict[str, Material]:
    finemet = Material("CAAC-630 (Finemet)", B_limit_T=1.00)  # user adjusts
    ferrite = Material("High-perm Ferrite", B_limit_T=0.20)
    steel = Material("Silicon Steel (low-f)", B_limit_T=1.00)
    return {"CAAC-630 (Finemet)": finemet, "High-perm Ferrite": ferrite, "Silicon Steel (low-f)": steel}
