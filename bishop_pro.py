import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from scipy.optimize import minimize

# --- CLASSES ---
class CoucheSol:
    def __init__(self, nom, y_top, y_bot, c, phi, gamma, color):
        self.nom, self.y_top, self.y_bot = nom, y_top, y_bot
        self.c, self.phi, self.gamma = c, math.radians(phi), gamma
        self.color = color

# --- MOTEUR DE CALCUL (Gardé intact) ---
# [Insère ici ta fonction calcul_bishop existante]

# --- FONCTION DE DESSIN PROFESSIONNELLE ---
def dessiner_stratigraphie(ax, couches, H, L, y_nappe):
    # Dessin des couches
    for c in couches:
        ax.fill_between([-L*0.2, L*1.2], c.y_bot, c.y_top, color=c.color, alpha=0.4, label=f"{c.nom} (c={c.c:.0f}, φ={math.degrees(c.phi):.0f}°, γ={c.gamma:.1f})")
    
    # Dessin du talus
    ax.plot([0, L], [0, H], 'k-', lw=2)
    ax.axhline(y_nappe, color='blue', lw=1.5, ls='--', label=f'Nappe (z={y_nappe}m)')
    ax.set_title("Modèle Stratigraphique", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')

# --- INTERFACE ---
# ... (Code de saisie des paramètres comme précédemment) ...

if col_btn1.button("👁️ Vérifier Stratigraphie"):
    obj_couches = get_couches()
    fig, ax = plt.subplots(figsize=(10, 5))
    dessiner_stratigraphie(ax, obj_couches, H, L, y_nappe)
    st.pyplot(fig)
    # Affichage tabulaire des données
    st.table([{'Couche': c.nom, 'C (kPa)': c.c, 'Phi (°)': math.degrees(c.phi), 'Gamma (kN/m3)': c.gamma} for c in obj_couches])

if col_btn2.button("🚀 Lancer l'Analyse"):
    with st.spinner("Analyse de stabilité en cours..."):
        obj_couches = get_couches()
        # Appel de l'optimisation
        res = minimize(calcul_bishop, [L/2, H*2, H*1.5], args=(obj_couches, H, L, y_nappe, q, kh, False), method='Nelder-Mead')
        fs_opt = res.fun
        xc, yc, R = res.x
        
        # Affichage résultat avec cercle et gradient
        fig, ax = plt.subplots(figsize=(10, 6))
        dessiner_stratigraphie(ax, obj_couches, H, L, y_nappe)
        
        # Cercle de rupture
        t = np.linspace(0, 2*np.pi, 100)
        ax.plot(xc + R*np.cos(t), yc + R*np.sin(t), 'r-', lw=3, label=f'Cercle Critique (Fs={fs_opt:.2f})')
        
        st.pyplot(fig)
        
        # Synthèse professionnelle
        st.subheader("Synthèse des résultats")
        if fs_opt < 1.0:
            st.error(f"### Instabilité critique détectée (Fs = {fs_opt:.2f})")
            st.write("Le facteur de sécurité est inférieur à 1. Le talus est considéré comme instable dans les conditions actuelles. Des mesures de confortement sont nécessaires.")
        elif fs_opt < 1.5:
            st.warning(f"### Stabilité limitée (Fs = {fs_opt:.2f})")
            st.write("Le talus est stable mais avec une marge de sécurité réduite. Une surveillance ou des travaux de drainage pourraient être envisagés.")
        else:
            st.success(f"### Stabilité assurée (Fs = {fs_opt:.2f})")
            st.write("Le talus présente un coefficient de sécurité satisfaisant selon les critères géotechniques usuels.")
