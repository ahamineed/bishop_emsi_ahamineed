import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from scipy.optimize import minimize

# --- 1. STRUCTURE ET MOTEUR DE CALCUL EXPERT ---
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

def calcul_bishop_expert(params, couches, H, L, y_nappe, q, kh, return_slices=False):
    xc, yc, R = params
    xs = np.linspace(0.05, L-0.05, 40)
    dx = xs[1] - xs[0]
    fs = 1.2
    slices_data = []
    for _ in range(15):
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
            u = max(0, (y_nappe - y_cercle) * 9.81) if y_cercle < y_nappe else 0
            m_alpha = math.cos(alpha) + (math.sin(alpha) * math.tan(phi_p)) / fs
            res_force = (c_p * dx + (W - u * dx) * math.tan(phi_p)) / max(0.01, m_alpha)
            m_r_total += res_force * R
            m_m_total += W * math.sin(alpha) * R + (kh * W * (yc - y_cercle))
            if return_slices:
                ratio = (res_force * R) / (abs(W * math.sin(alpha) * R) + 0.1)
                current_slices.append({'coords': [(x-dx/2, y_cercle), (x+dx/2, y_cercle), (x+dx/2, y_surf), (x-dx/2, y_surf)], 'ratio': ratio})
        if m_m_total <= 0: return (999, []) if return_slices else 999
        fs_new = m_r_total / m_m_total
        if abs(fs_new - fs) < 0.005: break
        fs = fs_new
        slices_data = current_slices
    return (fs, slices_data) if return_slices else fs

# --- 2. CONFIGURATION INTERFACE ---
st.set_page_config(page_title="GEO-STAB", layout="wide")
st.title("🏗️ GEO-STAB : Analyse de Stabilité des talus (Méthode de Bishop)")

def valider_parametres():
    erreurs = []
    if H <= 0: erreurs.append("La Hauteur H doit être positive.")
    if L <= 0: erreurs.append("La Largeur L doit être positive.")
    if q < 0: erreurs.append("La Surcharge q ne peut pas être négative.")
    if kh < 0: erreurs.append("Le coefficient kh ne peut pas être négatif.")
    for d in data_c:
        if d['ep'] <= 0: erreurs.append(f"L'épaisseur de '{d['nom']}' doit être positive.")
        if d['c'] < 0: erreurs.append(f"La cohésion de '{d['nom']}' ne peut pas être négative.")
        if d['ph'] < 0: erreurs.append(f"L'angle Phi de '{d['nom']}' ne peut pas être négatif.")
        if d['ga'] <= 0: erreurs.append(f"Le poids Gamma de '{d['nom']}' doit être positif.")
    return erreurs

with st.sidebar:
    st.header("📐 Géométrie & Conditions")
    H = st.number_input("Hauteur H (m)", value=10.0)
    L = st.number_input("Largeur L (m)", value=15.0)
    y_nappe = st.number_input("Niveau Nappe (m)", value=-2.0)
    q = st.number_input("Surcharge q (kPa)", value=0.0)
    kh = st.number_input("Séisme kh (g)", value=0.0)

st.header("🌍 Paramètres des Couches")
nb_c = st.number_input("Nombre de couches", 1, 6, 2)
data_c = []
colors_list = ['#8D6E63', '#D7CCC8', '#A1887F', '#BCAAA4', '#E0E0E0', '#BDBDBD']

cols_c = st.columns(nb_c)
for i in range(nb_c):
    with cols_c[i]:
        st.subheader(f"Couche {i+1}")
        nom_c = st.text_input(f"Nom", value=f"Couche {i+1}", key=f"n_{i}")
        ep = st.number_input("Épaisseur (m)", value=5.0, key=f"e_{i}")
        c = st.number_input("c' (kPa)", value=12.0, key=f"c_{i}")
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

# --- 3. ACTIONS ---
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
        
        ax.add_patch(Polygon([(-L, 0), (0, 0), (L, H), (L*2, H), (L*2, H+50), (-L, H+50)], facecolor='white', zorder=2))
        ax.plot([-L, 0, L, L*2], [0, 0, H, H], 'k-', lw=3, zorder=3)
        ax.axhline(y_nappe, color='#0277BD', ls='-.', lw=2, label="Nappe Phréatique", zorder=5)
        
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Altitude (m)")
        ax.set_aspect('equal')
        ax.set_xlim(-L*0.4, L*1.7)
        ax.set_ylim(-20, H+10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Propriétés des Sols")
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader("📋 Récapitulatif technique")
        st.table([{"Désignation": c.nom, "c' (kPa)": c.c, "φ' (°)": c.phi_deg, "γ (kN/m³)": c.gamma} for c in couches])

if col_btn2.button("🚀 Lancer l'Analyse Expert"):
    erreurs = valider_parametres()
    if erreurs:
        for err in erreurs: st.error(err)
    else:
        with st.spinner("Recherche de la surface critique..."):
            couches = preparer_couches()
            best_fs, best_params, best_slices = 999, None, []
            for xc in np.linspace(-L*0.3, L*1.3, 10):
                for yc in np.linspace(H*1.1, H*2.5, 10):
                    R = math.sqrt((xc - 0)**2 + (yc - 0)**2) * 1.15
                    fs, tranches = calcul_bishop_expert((xc, yc, R), couches, H, L, y_nappe, q, kh, return_slices=True)
                    if 0.5 < fs < best_fs:
                        best_fs, best_params, best_slices = fs, (xc, yc, R), tranches

            fig, ax = plt.subplots(figsize=(14, 8))
            for c in couches:
                ax.fill_between([-L, L*2], c.y_bot, c.y_top, color=c.color, alpha=0.4, 
                                label=f"{c.nom} (c'={c.c}kPa, φ'={c.phi_deg}°)")
            
            ax.add_patch(Polygon([(-L, 0), (0, 0), (L, H), (L*2, H), (L*2, H+50), (-L, H+50)], facecolor='white', zorder=2))
            ax.plot([-L, 0, L, L*2], [0, 0, H, H], 'k-', lw=3, zorder=3)

            cmap, norm = plt.get_cmap('RdYlGn'), plt.Normalize(0.6, 1.4)
            for t in best_slices:
                ax.add_patch(Polygon(t['coords'], facecolor=cmap(norm(t['ratio'])), edgecolor='black', lw=0.4, alpha=0.8, zorder=4))

            if best_params:
                xc, yc, R = best_params
                th = np.linspace(1.1*np.pi, 1.9*np.pi, 150)
                ax.plot(xc + R*np.cos(th), yc + R*np.sin(th), 'r--', lw=3, label=f"Cercle Critique (Fs={best_fs:.3f})")

            ax.axhline(y_nappe, color='#0277BD', ls='-.', lw=2, label="Nappe Phréatique", zorder=5)
            
            # --- COLORBAR POSITIONNÉE EN BAS POUR ÉVITER LE CHEVAUCHEMENT ---
            sm = ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.15)
            cbar.set_label("Gradient de Stabilité Locale (Tranches)", fontsize=10)
            
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Altitude (m)")
            ax.set_aspect('equal')
            ax.set_xlim(-L*0.4, L*1.7)
            ax.set_ylim(-20, H*1.5)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            
            plt.title(f"ANALYSE DE STABILITÉ - MÉTHODE DE BISHOP\nFacteur de Sécurité Global : {best_fs:.3f}", fontweight='bold')
            st.pyplot(fig)

            # --- INTERPRÉTATION DES RÉSULTATS ---
            st.divider()
            st.subheader("📝 Interprétation de la Stabilité")
            col_res1, col_res2 = st.columns([1, 2])
            col_res1.metric("Facteur de Sécurité (Fs)", f"{best_fs:.3f}")
            
            if best_fs < 1.0:
                col_res2.error(f"### ❌ RUPTURE DÉTECTÉE (Fs = {best_fs:.3f})")
                col_res2.write("Le massif est instable. Le glissement est imminent ou déjà effectif sous ces conditions. Un confortement immédiat (clouage, mur de soutènement) est nécessaire.")
            elif best_fs < 1.5:
                col_res2.warning(f"### ⚠️ STABILITÉ LIMITE (Fs = {best_fs:.3f})")
                col_res2.write("Le talus est stable mais avec une marge de sécurité insuffisante selon les normes courantes (souvent Fs > 1.5). Une attention particulière à l'eau est recommandée.")
            else:
                col_res2.success(f"### ✅ STABILITÉ ASSURÉE (Fs = {best_fs:.3f})")
                col_res2.write("Les conditions de sécurité sont satisfaisantes. Le talus est stable vis-à-vis d'une rupture circulaire selon les paramètres saisis.")
