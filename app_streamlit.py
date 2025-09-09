
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os, json

from mag_core import (
    MU0, CoreGeom, Material,
    build_voltage_knots, enforce_volt_second_balance,
    current_from_v_knots, analyze_current_dense, B_from_v_knots,
    core_library, material_library
)

st.set_page_config(page_title="Magnetics Designer - Inductor", layout="wide")
st.title("Magnetics Designer - Inductor (Steps 0–2)")
# --- Streamlit rerun helper (works for 1.30+ and older) ---
def _st_rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        _st_rerun()


# ---------------- Persistence helpers ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESIGN_DIR = os.path.join(BASE_DIR, "designs")
LAST_PATH = os.path.join(DESIGN_DIR, "_last.json")
os.makedirs(DESIGN_DIR, exist_ok=True)

DEFAULTS = {
    "L_target_uH": 200.0,
    "rows_saved": [
        {"Voltage (V)": 800.0, "Hold (us)": 15.0},
        {"Voltage (V)": 0.0,   "Hold (us)": 25.0},
        {"Voltage (V)": -800.0,"Hold (us)": 15.0},
    ],
    "t_trans_us": 0.0,
    "enforce": False,
    "Idc": 200.0,
    "use_i0": False,
    "core_choice": "CACC-630 (nanocrystalline C-core)",
    "Ae_custom": 1.558e-3,
    "le_custom": 0.323,
    "Ve_custom": float(1.558e-3*0.323),
    "mat_name": "CAAC-630 (Finemet)",
    "B_max_user": 1.000,
    "N": 25,
    "design_name": "last_design",
}

def load_design(path):
    try:
        with open(path, "r") as f:
            d = json.load(f)
        if isinstance(d, dict) and ("rows" in d) and ("rows_saved" not in d):
            d["rows_saved"] = d["rows"]
            d.pop("rows", None)
        return d
    except Exception:
        return None

def save_design(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def apply_design_to_state(d):
    keymap = {
        'design_name':'design_name_saved', 'rows':'rows_saved', 'rows_saved':'rows_saved',
        'L_target_uH':'L_target_uH_saved',
        't_trans_us':'t_trans_us_saved', 'enforce':'enforce_saved', 'Idc':'Idc_saved', 'use_i0':'use_i0_saved',
        'core_choice':'core_choice_saved', 'Ae_custom':'Ae_custom_saved', 'le_custom':'le_custom_saved', 'Ve_custom':'Ve_custom_saved',
        'mat_name':'mat_name_saved', 'B_max_user':'B_max_user_saved', 'N':'N_saved'
    }
    for k, v in d.items():
        st.session_state[keymap.get(k, k)] = v

# On first run, auto-load last if present
if not st.session_state.get("_loaded_once", False):
    d = load_design(LAST_PATH)
    if d is None:
        d = DEFAULTS.copy()
    apply_design_to_state(d)
    st.session_state["_loaded_once"] = True

# ---------------- Sidebar: Save/Load ----------------
with st.sidebar:
    st.header("Design persistence")
    design_name = st.text_input("Design name", value=st.session_state.get("design_name_saved","last_design"), key="design_name")
    st.session_state["design_name_saved"] = design_name

    # List saved designs
    saved = [fn[:-5] for fn in os.listdir(DESIGN_DIR) if fn.endswith(".json") and fn != "_last.json"]
    load_pick = st.selectbox("Load design", saved, index=0 if saved else None)

    cols = st.columns(3)
    with cols[0]:
        if st.button("Save"):
            st.session_state["_save_named"] = True
    with cols[1]:
        if st.button("Load") and load_pick:
            d = load_design(os.path.join(DESIGN_DIR, load_pick + ".json"))
            if d:
                apply_design_to_state(d)
                _st_rerun()
    with cols[2]:
        if st.button("New design"):
            apply_design_to_state(DEFAULTS.copy())
            _st_rerun()

    exp = st.checkbox("Enable export/import")
    if exp:
        st.caption("Export current design to JSON or import one.")
        st.session_state["_export_requested"] = True
        up = st.file_uploader("Import JSON", type=["json"])
        if up is not None:
            try:
                d = json.load(up)
                apply_design_to_state(d)
                _st_rerun()
            except Exception:
                st.error("Invalid JSON.")

# ---------------- Step 0: L target ----------------
st.header("Step 0 - Select Inductance Target")
L_target_uH = st.number_input(
    "Target inductance L [uH]",
    min_value=0.01, max_value=1_000_000.0,
    value=st.session_state.get("L_target_uH_saved", DEFAULTS["L_target_uH"]),
    step=1.0, format="%.3f", key="L_target_uH"
)
st.session_state["L_target_uH_saved"] = L_target_uH
L_target = L_target_uH * 1e-6

st.divider()

# ---------------- Step 1: Waveform Builder ----------------
st.header("Step 1 - Waveform Builder (Hold -> Transition -> Hold)")
st.caption("Each row is a HOLD (constant V for its duration). A TRANSITION is inserted when V changes.")

initial_rows = st.session_state.get("rows_saved", DEFAULTS["rows_saved"])
rows = st.data_editor(initial_rows, num_rows="dynamic", use_container_width=True, key="rows")
st.session_state["rows_saved"] = rows

t_trans_us = st.number_input(
    "Transition duration between rows [us]",
    min_value=0.0, max_value=1000.0,
    value=st.session_state.get("t_trans_us_saved", DEFAULTS["t_trans_us"]),
    step=0.1, format="%.2f", key="t_trans_us"
)
st.session_state["t_trans_us_saved"] = t_trans_us
t_trans = t_trans_us * 1e-6

enforce = st.checkbox(
    "Enforce volt-second balance (adjust last HOLD duration if possible)",
    value=st.session_state.get("enforce_saved", DEFAULTS["enforce"]), key="enforce"
)
st.session_state["enforce_saved"] = enforce

# DC bias
Idc = st.number_input(
    "DC bias current [A]", min_value=-1e6, max_value=1e6,
    value=st.session_state.get("Idc_saved", DEFAULTS["Idc"]),
    step=1.0, format="%.3f", key="Idc"
)
st.session_state["Idc_saved"] = Idc

use_i0 = st.checkbox(
    "Use initial current i0 for DC bias (instead of offset)",
    value=st.session_state.get("use_i0_saved", DEFAULTS["use_i0"]), key="use_i0"
)
st.session_state["use_i0_saved"] = use_i0

# Parse rows
parsed = []
for r in rows:
    try:
        V = float(r["Voltage (V)"])
        dt = float(r["Hold (us)"]) * 1e-6
        if dt > 0:
            parsed.append((V, dt))
    except Exception:
        pass

if len(parsed) < 1:
    st.error("Please enter at least one HOLD row with positive duration.")
    st.stop()

if enforce:
    parsed = enforce_volt_second_balance(parsed, t_trans)

# Build v(t) knots and current
t_v, v_knots = build_voltage_knots(parsed, t_trans)
periodic = abs(np.trapz(v_knots, t_v)) < 1e-12

if use_i0:
    td, i_raw = current_from_v_knots(L_target, t_v, v_knots, i0=Idc, periodic=periodic)
    i_total = i_raw
else:
    td, i_raw = current_from_v_knots(L_target, t_v, v_knots, i0=0.0, periodic=periodic)
    i_total = i_raw + Idc

# Zero-mean AC
T_per = td[-1] if td[-1] > 0 else 1.0
Iavg_total = float(np.trapz(i_total, td) / T_per)
i_ac = i_total - Iavg_total

# Plots
c1, c2 = st.columns(2)
with c1:
    st.subheader("Voltage Waveform")
    fig, ax = plt.subplots()
    ax.plot(t_v*1e6, v_knots)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("v(t) [V]")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with c2:
    st.subheader("Current Waveform (AC + DC bias)")
    fig2, ax2 = plt.subplots()
    ax2.plot(td*1e6, i_total, label="i_total")
    ax2.plot(td*1e6, i_ac, linestyle="--", label="i_ac (zero-mean)")
    ax2.set_xlabel("Time [us]")
    ax2.set_ylabel("i(t) [A]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

# Stats metrics
stats = analyze_current_dense(td, i_total)
T_us = stats['T_s']*1e6
f_kHz = stats['f_Hz']/1e3
Irms = stats['Irms_A']
Ipk = stats['Ipk_A']
Ipp = stats['Ipp_A']
Iavg = Iavg_total
colA, colB, colC = st.columns(3)
with colA:
    st.metric('T [us]', f'{T_us:.3f}')
    st.metric('f [kHz]', f'{f_kHz:.3f}')
with colB:
    st.metric('I_rms [A]', f'{Irms:.3f}')
    st.metric('I_avg [A]', f'{Iavg:.3f}')
with colC:
    st.metric('I_pk [A]', f'{Ipk:.3f}')
    st.metric('I_pp [A]', f'{Ipp:.3f}')

st.divider()

# ---------------- Step 2: Core, N, gap, B(t) ----------------
st.header("Step 2 - Core Selection, Suggested Turns & Airgap, and Flux Density")

cores = core_library()
core_options = list(cores.keys()) + ["Custom"]
core_choice_default = st.session_state.get("core_choice_saved", DEFAULTS["core_choice"])
core_index = core_options.index(core_choice_default) if core_choice_default in core_options else 0
core_choice = st.selectbox("Core", core_options, index=core_index, key="core_choice")
st.session_state["core_choice_saved"] = core_choice

if core_choice == "Custom":
    Ae = st.number_input(
        "Ae (m^2)", min_value=1e-8, max_value=1e-2,
        value=st.session_state.get("Ae_custom_saved", DEFAULTS["Ae_custom"]),
        step=1e-6, format="%.6e", key="Ae_custom"
    )
    st.session_state["Ae_custom_saved"] = Ae

    le = st.number_input(
        "le (m)", min_value=1e-3, max_value=1.0,
        value=st.session_state.get("le_custom_saved", DEFAULTS["le_custom"]),
        step=1e-3, format="%.6f", key="le_custom"
    )
    st.session_state["le_custom_saved"] = le

    Ve = st.number_input(
        "Ve (m^3)", min_value=1e-9, max_value=1e-1,
        value=st.session_state.get("Ve_custom_saved", DEFAULTS["Ve_custom"]),
        step=1e-8, format="%.6e", key="Ve_custom"
    )
    st.session_state["Ve_custom_saved"] = Ve

    core = CoreGeom("Custom", Ae, le, Ve)
else:
    core = cores[core_choice]
    st.caption(f"Preset core: {core.name} - Ae={core.Ae_m2:.3e} m^2, le={core.le_m:.3f} m, Ve~{core.Ve_m3:.3e} m^3")

materials = material_library()
mat_names = list(materials.keys())
mat_choice_default = st.session_state.get("mat_name_saved", DEFAULTS["mat_name"])
mat_idx = mat_names.index(mat_choice_default) if mat_choice_default in mat_names else 0
mat_name = st.selectbox("Material", mat_names, index=mat_idx, key="mat_name")
st.session_state["mat_name_saved"] = mat_name
material = materials[mat_name]

B_max_user = st.number_input(
    "Material B_max [T]", min_value=0.05, max_value=2.50,
    value=st.session_state.get("B_max_user_saved", DEFAULTS["B_max_user"]),
    step=0.01, format="%.3f", key="B_max_user"
)
st.session_state["B_max_user_saved"] = B_max_user

# Suggest N from Ipk
Ipk_est = float(np.max(np.abs(i_total)))
N_suggest_calc = int(max(1, np.ceil((L_target * Ipk_est) / (core.Ae_m2 * B_max_user))))

# User-select N
N = st.number_input(
    "Turns N (suggested)", min_value=1, max_value=5000,
    value=st.session_state.get("N_saved", N_suggest_calc),
    step=1, key="N"
)
st.session_state["N_saved"] = N

# Airgap metric + warning
lg = MU0 * core.Ae_m2 * (N**2) / L_target
lg_mm = float(lg*1e3)
col_gap = st.columns(3)[1]
with col_gap:
    st.metric("Airgap lg [mm]", f"{lg_mm:.3f}")
    if lg_mm > 10.0:
        st.warning(f"Computed airgap is {lg_mm:.1f} mm (>10 mm). Consider reducing N, increasing Ae, or changing L.")
st.metric("Material B_max [T]", f"{B_max_user:.3f}")

# Flux from current
coef_B = (L_target / (N * core.Ae_m2))
B_total = coef_B * i_total
B_ac_plot = coef_B * i_ac
Bpk = float(np.max(np.abs(B_total)))

st.subheader("Flux Density")
fig3, ax3 = plt.subplots()
ax3.plot(td*1e6, B_total, label="B_total (from i)")
ax3.plot(td*1e6, B_ac_plot, linestyle="--", label="B_ac (from i)")
ax3.axhline(B_max_user, linestyle="--", label="B_max")
ax3.axhline(-B_max_user, linestyle="--")
ax3.set_xlabel("Time [us]")
ax3.set_ylabel("B [T]")
ax3.grid(True, alpha=0.3)
ax3.legend()
st.pyplot(fig3); plt.close(fig3)

st.metric("B_pk [T]", f"{Bpk:.4f}")

# ---- Calculations & Equations (optional) ----
with st.expander("Show calculations & equations", expanded=False):
    mu0 = MU0
    Ae_val = core.Ae_m2
    N_int = int(N)
    coef_B_local = L_target/(N*Ae_val)
    Ipk_calc = float(np.max(np.abs(i_total)))
    Ipp_calc = float(np.max(i_total) - np.min(i_total))
    st.latex(r"B(t) = \frac{L}{N\,A_e}\,i(t)")
    st.latex(r"B_{pk} = \frac{L}{N\,A_e}\,I_{pk}")
    st.write("Numbers: B_pk = (L/(N*Ae))*I_pk = ({:.3e}/({}*{:,.3e})) * {:.3f} = **{:.3f} T**".format(
        L_target, N_int, Ae_val, Ipk_calc, coef_B_local*Ipk_calc
    ))
    if lg > 0:
        st.latex(r"k_B \equiv \frac{dB}{di} \approx \frac{\mu_0 N}{\ell_g}")
        st.write("Numbers: k_B ≈ (mu0*N)/lg = ({:.3e}*{})/{:.4e} = {:.3e} T/A".format(mu0, N_int, lg, mu0*N/lg))
    try:
        tB_vs, B_vs = B_from_v_knots(core, N_int, t_v, v_knots, center_zero=True)
        dBpp_vs = float(np.max(B_vs) - np.min(B_vs))
        st.write("ΔB_pp (from v) ≈ {:.3f} T".format(dBpp_vs))
    except Exception:
        st.caption("Could not compute B from v(t) for cross-check.")
    st.write("I_pk = {:.3f} A,  I_pp = {:.3f} A,  I_avg = {:.3f} A".format(Ipk_calc, Ipp_calc, Iavg))
    Nsug_formula = (L_target*Ipk_calc)/(Ae_val*B_max_user)
    st.write("N_suggest = ceil( L*I_pk / (Ae*B_max) ) = ceil( {:.3e} * {:.3f} / ({:.3e} * {:.3f}) ) = **{}**".format(
        L_target, Ipk_calc, Ae_val, B_max_user, int(max(1, np.ceil(Nsug_formula)))
    ))

st.divider()
st.header("Step 3 — Loss & Winding Structure (coming next)")

# ---------------- After UI: Auto-save last & optional save/export ----------------
current_design = {
    "L_target_uH": float(st.session_state.get('L_target_uH_saved', 200.0)),
    "rows_saved": st.session_state.get('rows_saved', []),
    "t_trans_us": float(st.session_state.get('t_trans_us_saved', 0.0)),
    "enforce": bool(st.session_state.get('enforce_saved', False)),
    "Idc": float(st.session_state.get('Idc_saved', 0.0)),
    "use_i0": bool(st.session_state.get('use_i0_saved', False)),
    "core_choice": st.session_state.get('core_choice_saved', 'ETD49'),
    "Ae_custom": float(st.session_state.get('Ae_custom_saved', 1.558e-3)),
    "le_custom": float(st.session_state.get('le_custom_saved', 0.323)),
    "Ve_custom": float(st.session_state.get('Ve_custom_saved', 1.558e-3*0.323)),
    "mat_name": st.session_state.get('mat_name_saved', 'CAAC-630 (Finemet)'),
    "B_max_user": float(st.session_state.get('B_max_user_saved', 1.0)),
    "N": int(st.session_state.get('N_saved', 25)),
    "design_name": st.session_state.get('design_name_saved', 'last_design'),
}

with open(LAST_PATH, "w") as f:
    json.dump(current_design, f, indent=2)

if st.session_state.get("_save_named"):
    name = st.session_state.get("design_name_saved","design")
    with open(os.path.join(DESIGN_DIR, name + ".json"), "w") as f:
        json.dump(current_design, f, indent=2)
    st.session_state["_save_named"] = False
    st.success(f"Saved design '{name}'.")

if st.session_state.get("_export_requested"):
    st.sidebar.download_button(
        "Export current design JSON",
        data=json.dumps(current_design, indent=2),
        file_name=f"{st.session_state.get('design_name_saved','design')}.json",
        mime="application/json"
    )
