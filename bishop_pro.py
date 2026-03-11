import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from scipy.optimize import minimize

# --- CLASSES ET CALCUL BISHOP ---
class CoucheSol:
    def __init__(self, nom, y_top, y_bot, c, phi, gamma, color):
        self.nom, self.y_top, self.y_bot = nom, y_top, y_bot
        self.c, self.phi, self.gamma = c, math.radians(phi), gamma
        self.color = color

def calcul_bishop(params, couches, H, L, y_nappe, q, kh, return_slices=False):
    xc, yc, R = params
    n_tranches = 45
    x_min_c, x_max_c = xc - R, xc + R
    x_start, x_end = max(x_min_c + 0.1, -L*0.2), min(x_max_c - 0.1, L*1.3)
    if x_end <= x_start: return 10.0
    xs = np.linspace(x_start, x_end, n_tranches)
    dx = xs[1] - xs[0]
    fs = 1.2
    for _ in range(20):
        m_r, m_m, slices_data, valid = 0, 0, [], 0
        for x in xs:
            val = R**2 - (x - xc)**2
            if val < 0: continue
            y_c = yc - math.sqrt(val)
            y_s = 0 if x < 0 else (H/L)*x if x < L else H
            if y_s <= y_c + 0.05: continue
            c_p, phi_p, p_col = 0, 0, 0
            for c in couches:
                if y_c >= c.y_bot and y_c <= c.y_top:
                    c_p, phi_p = c.c, c.phi
                    break
            y_curr = y_s
            for c in couches:
                top, bot = min(y_curr, c.y_top), max(y_c, c.y_bot)
                if top > bot: p_col += (top - bot) * c.gamma
            W = p_col * dx + (q * dx if x > L else 0)
            u = max(0, (y_nappe - y_c) * 9.81) if y_c < y_nappe else 0
            alpha = math.atan((x - xc) / max(0.1, abs(y_c - yc)))
            m_alpha = math.cos(alpha) + (math.sin(alpha) * math.tan(phi_p)) / max(0.1, fs)
            res = (c_p * dx + (W - u * dx) * math.tan(phi_p)) / max(0.01, m_alpha)
            m_r += res * R
            m_m += W * math.sin(alpha) * R + (kh * W * (yc - (y_s + y_c)/2))
            valid += 1
            if return_slices:
                slices_data.append({'coords': [(x-dx/2, y_c), (x+dx/2, y_c), (x+dx/2, y_s), (x-dx/2, y_s)], 'ratio': res*R/(abs(W*math.sin(alpha)*R)+0.1)})
        if valid < 5 or m_m <= 0: return 10.0
        new_fs = m_r / m_m
        if abs(new_fs - fs) < 0.002: break
        fs = new_fs
    return (fs, slices_data) if return_slices else fs

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="GeoStab Pro", layout="wide")
st.title("🚀 GEO-STAB PRO : Analyse de Stabilité")

st.sidebar.header("Paramètres Globaux")
H = st.sidebar.number_input("Hauteur H (m)", value=10.0)
L = st.sidebar.number_input("Largeur L (m)", value=20.0)
y_nappe = st.sidebar.number_input("Niveau Nappe (m)", value=-5.0)
q = st.sidebar.number_input("Surcharge q (kPa)", value=0.0)
kh = st.sidebar.number_input("Séisme kh (g)", value=0.0)

st.header("Stratigraphie")
nb_couches = st.number_input("Nombre de couches", min_value=1, max_value=4, value=1)
couches_data = []
cols = st.columns(nb_couches)
colors = ['#FAD7A0', '#E59866', '#D4AC0D', '#A04000']
for i in range(nb_couches):
    with cols[i]:
        st.subheader(f"Couche {i+1}")
        ep = st.number_input(f"Épaisseur (m)", value=5.0, key=f"ep_{i}")
        c = st.number_input(f"Cohésion (kPa)", value=20.0, key=f"c_{i}")
        phi = st.number_input(f"Phi (°)", value=25.0, key=f"phi_{i}")
        gamma = st.number_input(f"Gamma (kN/m³)", value=18.0, key=f"gamma_{i}")
        couches_data.append({'ep': ep, 'c': c, 'phi': phi, 'gamma': gamma, 'color': colors[i]})

# Logique de construction des objets
def get_couches():
    obj_couches = []
    y_act = H
    for data in couches_data:
        obj_couches.append(CoucheSol("C", y_act, y_act - data['ep'], data['c'], data['phi'], data['gamma'], data['color']))
        y_act -= data['ep']
    return obj_couches

col_btn1, col_btn2 = st.columns(2)

if col_btn1.button("👁️ Afficher Stratigraphie"):
    obj_couches = get_couches()
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in obj_couches:
        ax.fill_between([-L, L*2], c.y_bot, c.y_top, color=c.color, alpha=0.5)
    st.pyplot(fig)

if col_btn2.button("🚀 Lancer l'Analyse"):
    with st.spinner("Calcul en cours..."):
        obj_couches = get_couches()
        fs = calcul_bishop([L/2, H*2, H*1.5], obj_couches, H, L, y_nappe, q, kh)
        st.metric("Facteur de Sécurité (Fs)", f"{fs:.3f}")
        if fs < 1.0: st.error("Le talus est instable !")
        else: st.success("Le talus est stable.")
