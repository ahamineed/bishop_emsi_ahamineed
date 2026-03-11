import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from scipy.optimize import minimize

# --- 1. STRUCTURE ET MOTEUR DE CALCUL BISHOP ---
class CoucheSol:
    def __init__(self, nom, y_top, y_bot, c, phi, gamma, color):
        self.nom = nom
        self.y_top, self.y_bot = y_top, y_bot
        self.c = c
        self.phi_deg = phi
        self.phi = math.radians(phi)
        self.gamma, self.color = gamma, color

def get_parametres_expert(y_m, y_surf, couches):
    c_base, phi_base, poids_total = 0, 0, 0
    for c in couches:
        if y_m >= c.y_bot and y_m <= c.y_top:
            c_base, phi_base = c.c, c.phi
            break
    y_curr = y_surf
    for c in couches:
        top, bot = min(y_curr, c.y_top), max(y_m, c.y_bot)
        if top > bot: poids_total += (top - bot) * c.gamma
    return c_base, phi_base, poids_total

def calcul_bishop_expert(params, couches, H, L, y_nappe, nappe_active, q, kh, return_slices=False):
    xc, yc, R = params
    xs = np.linspace(0.05, L-0.05, 45) # Plus de tranches pour plus de précision
    dx = xs[1] - xs[0]
    fs = 1.2
    slices_data = []

    for _ in range(25): # Augmentation des itérations pour convergence difficile
        m_r_total, m_m_total = 0, 0
        current_slices = []
        for x in xs:
            inter = R**2 - (x - xc)**2
            if inter < 0: continue
            y_cercle = yc - math.sqrt(inter)
            y_surf = (H/L)*x if 0 <= x <= L else (0 if x < 0 else H)
            
            if y_surf <= y_cercle + 0.05: continue
            
            alpha = math.atan((x - xc) / max(0.1, abs(y_cercle - yc)))
            c_p, phi_p, p_col = get_parametres_expert(y_cercle, y_surf, couches)
            
            W = p_col * dx + (q * dx if x > L else 0)
            # Gestion de la pression interstitielle u
            u = max(0, (y_nappe - y_cercle) * 9.81) if (nappe_active and y_cercle < y_nappe) else 0

            m_alpha = math.cos(alpha) + (math.sin(alpha) * math.tan(phi_p)) / fs
            res_force = (c_p * dx + (W - u * dx) * math.tan(phi_p)) / max(0.01, m_alpha)

            m_r_total += res_force * R
            m_m_total += W * math.sin(alpha) * R + (kh * W * (yc - (y_surf + y_cercle)/2))

            if return_slices:
                # Ratio de stabilité local pour le gradient
                ratio = (res_force * R) / (abs(W * math.sin(alpha) * R) + 0.1)
                current_slices.append({'coords': [(x-dx/2, y_cercle), (x+dx/2, y_cercle), (x+dx/2, y_surf), (x-dx/2, y_surf)], 'ratio': ratio})

        if m_m_total <= 0: return (9.99, []) if return_slices else 9.99
        fs_new = m_r_total / m_m_total
        if abs(fs_new - fs) < 0.002: break
        fs = fs_new
        slices_data = current_slices
    
    return (fs, slices_data) if return_slices else fs

# --- 2. INTERFACE UTILISATEUR ---
st.set_page_config(page_title="GEO-STAB", layout="wide")
st.title("🏗️ GEO-STAB : Analyse de Stabilité des talus (Méthode de Bishop)")

# Validation des entrées
def valider_parametres():
    erreurs = []
    if H <= 0 or L <= 0: erreurs.append("Dimensions H et L doivent être strictement positives.")
    if q < 0 or kh < 0: erreurs.append("La surcharge et le séisme ne peuvent pas être négatifs.")
    for d in data_c:
        if d['ep'] <= 0: erreurs.append(f"Épaisseur de '{d['nom']}' invalide.")
        if d['c'] < 0 or d['ph'] < 0: erreurs.append(f"Paramètres c' ou φ' négatifs pour '{d['nom']}'.")
    return erreurs

with st.sidebar:
    st.header("📐 Paramètres du Talus")
    H = st.number_input("Hauteur H (m)", value=10.0, min_value=0.1)
    L = st.number_input("Largeur L (m)", value=15.0, min_value=0.1)
    
    st.divider()
    nappe_active = st.checkbox("Présence de la nappe phréatique", value=False)
    y_nappe = -50.0 # Valeur par défaut (profonde)
    if nappe_active:
        y_nappe = st.number_input("Niveau de la nappe (m)\n(Niveau 0 = Pied du talus)", value=-2.0)
    
    st.divider()
    st.subheader("Sollicitations")
    q = st.number_input("Surcharge q (kPa)", value=0.0)
    kh = st.number_input("Séisme kh (g)", value=0.0, step=0.05)

st.header("🌍 Stratigraphie")
nb_c = st.number_input("Nombre de couches géologiques", 1, 6, 2)
data_c = []
colors_list = ['#8D6E63', '#D7CCC8', '#A1887F', '#BCAAA4', '#E0E0E0', '#BDBDBD']

cols_c = st.columns(nb_c)
for i in range(nb_c):
    with cols_c[i]:
        st.subheader(f"Couche {i+1}")
        nom_c = st.text_input(f"Nom", value=f"Couche {i+1}", key=f"n_{i}")
        ep = st.number_input("Épaisseur (m)", value=5.0, key=f"e_{i}")
        c = st.number_input("c' (kPa)", value=15.0, key=f"c_{i}")
        ph = st.number_input("φ' (°)", value=25.0, key=f"p_{i}")
        ga = st.number_input("γ (kN/m³)", value=19.0, key=f"g_{i}")
        data_c.append({'nom': nom_c, 'ep': ep, 'c': c, 'ph': ph, 'ga': ga, 'color': colors_list[i % 6]})

def preparer_couches():
    couches = []
    y_top = H
    for d in data_c:
        couches.append(CoucheSol(d['nom'], y_top, y_top-d['ep'], d['c'], d['ph'], d['ga'], d['color']))
        y_top -= d['ep']
    couches[-1].y_bot = -40 
    return couches

# --- 3. LOGIQUE D'AFFICHAGE ---
col_btn1, col_btn2 = st.columns(2)

if col_btn1.button("👁️ Afficher la Stratigraphie"):
    erreurs = valider_parametres()
    if erreurs:
        for err in erreurs: st.error(err)
    else:
        couches = preparer_couches()
        fig, ax = plt.subplots(figsize=(12, 7))
        for c in couches:
            ax.fill_between([-L, L*2], c.y_bot, c.y_top, color=c.color, alpha=0.5, 
                            label=f"{c.nom} (c'={c.c}kPa, φ'={c.phi_deg}°)")
        
        # Masque Blanc (Ciel)
        ax.add_patch(Polygon([(-L, 0), (0, 0), (L, H), (L*2, H), (L*2, H+50), (-L, H+50)], facecolor='white', zorder=2))
        ax.plot([-L, 0, L, L*2], [0, 0, H, H], 'k-', lw=3, zorder=3)
        
        if nappe_active:
            ax.axhline(y_nappe, color='#0277BD', ls='-.', lw=2, label=f"Nappe ({y_nappe}m)", zorder=5)
        
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Altitude (m)")
        ax.set_aspect('equal')
        ax.set_xlim(-L*0.4, L*1.7)
        ax.set_ylim(-20, H+10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Propriétés Mécaniques")
        plt.tight_layout()
        st.pyplot(fig)

if col_btn2.button("🚀 Lancer l'Analyse"):
    erreurs = valider_parametres()
    if erreurs:
        for err in erreurs: st.error(err)
    else:
        with st.spinner("Optimisation du cercle de rupture..."):
            couches = preparer_couches()
            best_fs, best_params, best_slices = 9.99, None, []
            
            # Grille de recherche pour initialiser l'optimisation
            for xc in np.linspace(-L*0.2, L*1.2, 12):
                for yc in np.linspace(H*1.1, H*2.2, 12):
                    R = math.sqrt((xc - 0)**2 + (yc - 0)**2) * 1.1
                    fs, tranches = calcul_bishop_expert((xc, yc, R), couches, H, L, y_nappe, nappe_active, q, kh, return_slices=True)
                    if 0.6 < fs < best_fs:
                        best_fs, best_params, best_slices = fs, (xc, yc, R), tranches

            fig, ax = plt.subplots(figsize=(14, 8))
            for c in couches:
                ax.fill_between([-L, L*2], c.y_bot, c.y_top, color=c.color, alpha=0.3, label=f"{c.nom}")
            
            # Masque Blanc
            ax.add_patch(Polygon([(-L, 0), (0, 0), (L, H), (L*2, H), (L*2, H+50), (-L, H+50)], facecolor='white', zorder=2))
            ax.plot([-L, 0, L, L*2], [0, 0, H, H], 'k-', lw=3, zorder=3)

            # Gradient de stabilité (Tranches)
            cmap, norm = plt.get_cmap('RdYlGn'), plt.Normalize(0.6, 1.4)
            for t in best_slices:
                ax.add_patch(Polygon(t['coords'], facecolor=cmap(norm(t['ratio'])), edgecolor='black', lw=0.3, alpha=0.8, zorder=4))

            if best_params:
                xc, yc, R = best_params
                th = np.linspace(1.05*np.pi, 1.95*np.pi, 200)
                ax.plot(xc + R*np.cos(th), yc + R*np.sin(th), 'r--', lw=3, label=f"Cercle Critique (Fs={best_fs:.3f})")

            if nappe_active:
                ax.axhline(y_nappe, color='#0277BD', ls='-.', lw=2.5, label="Nappe Phréatique", zorder=5)
            
            # Colorbar horizontale
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.15)
            cbar.set_label("Ratio de Stabilité Local (Tranches)", fontsize=10)
            
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Altitude (m)")
            ax.set_aspect('equal')
            ax.set_xlim(-L*0.4, L*1.7)
            ax.set_ylim(-20, H*1.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            
            plt.title(f"ANALYSE DE STABILITÉ - MÉTHODE DE BISHOP\nFacteur de Sécurité Global : {best_fs:.3f}", fontweight='bold')
            st.pyplot(fig)

            # --- INTERPRÉTATION ---
            st.divider()
            st.subheader("📝 Diagnostic de l'Ingénieur")
            c1, c2 = st.columns([1, 2])
            c1.metric("F.S. Calculé", f"{best_fs:.3f}")
            
            if best_fs < 1.0:
                c2.error(f"### ❌ RUPTURE (Fs = {best_fs:.3f})")
                c2.write("Instabilité majeure. Le talus ne peut pas supporter son propre poids ou les surcharges actuelles.")
            elif best_fs < 1.5:
                c2.warning(f"### ⚠️ STABILITÉ PRÉCAIRE (Fs = {best_fs:.3f})")
                c2.write("Le talus est stable mais ne respecte pas les coefficients de sécurité réglementaires (généralement Fs > 1.5).")
            else:
                c2.success(f"### ✅ STABILITÉ CONFIRMÉE (Fs = {best_fs:.3f})")
                c2.write("Le massif présente une sécurité satisfaisante vis-à-vis d'une rupture circulaire.")
