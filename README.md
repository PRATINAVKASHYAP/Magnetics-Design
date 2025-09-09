# Magnetics Designer

An interactive **Streamlit-based tool** for magnetics design in power electronics.

## ✨ Features
- 🎯 **Target inductance calculator** (Vin, Vout, Iout, fsw, ripple%)
- 📈 **Waveform analysis** (buck, boost, buck-boost; ripple vs duty)
- ⚙️ **Parametric inductor design** (turns N, air gap g, Bmax)
- 🔋 **AC loss quick look** (Dowell-style estimate)
- 🧮 **Persistent equation editor** (JSON-backed, add/rename/delete)

## 🚀 Getting Started

```bash
# clone the repo
git clone https://github.com/PRATINAVKASHYAP/Magnetics-Design.git
cd Magnetics-Design

# create virtual environment (Python 3.13+ recommended)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# run app
python -m streamlit run app_streamlit.py
