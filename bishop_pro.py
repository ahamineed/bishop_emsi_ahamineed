import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from scipy.optimize import minimize

# --- 1. MOTEUR DE CALCUL (Bishop) ---
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

# --- 2. FONCTIONS D'AFFICHAGE ---
def dessiner_base(ax, couches, H, L, y_n):
    for c in couches: ax.fill_between([-L*0.2, L*1.2], c.y_bot, c.y_top, color=c.color, alpha=0.3, label=c.nom)
    ax.add_patch(Polygon([(-L*0.2,0), (0,0), (L,H), (L*1.2,H), (L*1.2,H*2), (-L*0.2,H*2)], facecolor='none', edgecolor='black', lw=2))
    ax.axhline(y_n, color='#2980b9', lw=2, ls='--', label='Niveau de la Nappe')
    ax.set_aspect('equal')

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="GeoStab Pro", layout="wide")
st.title("🏗️ GEO-STAB PRO : Analyse de Stabilité des Talus")

st.sidebar.header("Paramètres Géométrie")
H = st.sidebar.number_input("Hauteur H (m)", value=10.0)
L = st.sidebar.number_input("Largeur L (m)", value=20.0)
y_nappe = st.sidebar.number_input("Niveau Nappe (m)", value=-5.0)
q = st.sidebar.number_input("Surcharge q (kPa)", value=0.0)
kh = st.sidebar.number_input("Séisme kh (g)", value=0.0)

st.header("Stratigraphie")
nb_c = st.number_input("Nombre de couches", 1, 4, 1)
data_c = []
cols = st.columns(nb_c)
for i in range(nb_c):
    with cols[i]:
        st.subheader(f"Couche {i+1}")
        ep = st.number_input("Épaisseur (m)", value=5.0, key=f"e{i}")
        c = st.number_input("Cohésion (kPa)", value=20.0, key=f"c{i}")
        ph = st.number_input("Frottement (°)", value=25.0, key=f"p{i}")
        g = st.number_input("Gamma (kN/m³)", value=18.0, key=f"g{i}")
        data_c.append({'ep': ep, 'c': c, 'ph': ph, 'g': g})

def get_obj():
    l, y = [], H
    for d in data_c:
        l.append(CoucheSol("C", y, y-d['ep'], d['c'], d['ph'], d['g'], 'tan'))
        y -= d['ep']
    return l

c1, c2 = st.columns(2)
if c1.button("👁️ Vérifier Stratigraphie"):
    obj = get_obj()
    fig, ax = plt.subplots()
    dessiner_base(ax, obj, H, L, y_nappe)
    st.pyplot(fig)

if c2.button("🚀 Lancer l'Analyse"):
    with st.spinner("Calcul en cours..."):
        obj = get_obj()
        res = minimize(calcul_bishop, [L/2, H*2, H*1.5], args=(obj, H, L, y_nappe, q, kh, False), method='Nelder-Mead')
        fs, xc, yc, R = res.fun, *res.x
        _, slices = calcul_bishop([xc, yc, R], obj, H, L, y_nappe, q, kh, True)
        
        fig, ax = plt.subplots(figsize=(10,6))
        dessiner_base(ax, obj, H, L, y_nappe)
        t = np.linspace(0, 2*np.pi, 100)
        ax.plot(xc + R*np.cos(t), yc + R*np.sin(t), 'r-', lw=3, label=f'Rupture (Fs={fs:.2f})')
        st.pyplot(fig)
        
        st.metric("Facteur de Sécurité Optimal", f"{fs:.3f}")
        if fs < 1.0: st.error("⚠️ Talus Instable : Des mesures de confortement sont requises.")
        else: st.success("✅ Talus Stable : Le coefficient de sécurité est satisfaisant.")
