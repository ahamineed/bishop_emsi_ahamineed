import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from scipy.optimize import minimize

# --- STRUCTURE DES COUCHES ---
class CoucheSol:
    def __init__(self, nom, y_top, y_bot, c, phi, gamma, color):
        self.nom, self.y_top, self.y_bot = nom, y_top, y_bot
        self.c, self.phi, self.gamma = c, math.radians(phi), gamma
        self.color = color

# --- NOYAU DE CALCUL BISHOP ---
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
            
            y_centroid = (y_s + y_c) / 2
            m_r += res * R
            m_m += W * math.sin(alpha) * R + (kh * W * (yc - y_centroid))
            valid += 1
            if return_slices:
                slices_data.append({'coords': [(x-dx/2, y_c), (x+dx/2, y_c), (x+dx/2, y_s), (x-dx/2, y_s)], 'ratio': res*R/(abs(W*math.sin(alpha)*R)+0.1)})
        
        if valid < 5 or m_m <= 0: return 10.0
        new_fs = m_r / m_m
        if abs(new_fs - fs) < 0.002: break
        fs = new_fs
        
    return (fs, slices_data) if return_slices else fs

# --- INTERFACE GRAPHIQUE MODERNE (GUI) ---
class BishopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GEO-STAB PRO : Analyse de Stabilité des Talus")
        self.root.geometry("680x550")
        
        # Style global
        style = ttk.Style()
        style.theme_use('clam')
        self.root.configure(bg="#eceff1") # Gris très clair bleuté
        
        # Polices
        font_titre = ("Segoe UI", 14, "bold")
        font_label = ("Segoe UI", 10)
        
        # En-tête
        header = tk.Frame(root, bg="#2c3e50", pady=15)
        header.pack(fill="x")
        tk.Label(header, text="Moteur de Calcul BISHOP", fg="white", bg="#2c3e50", font=font_titre).pack()

        main_frame = tk.Frame(root, bg="#eceff1", padx=20, pady=10)
        main_frame.pack(fill="both", expand=True)

        # 1. GÉOMÉTRIE & CHARGEMENT (Alignés côte à côte)
        top_frame = tk.Frame(main_frame, bg="#eceff1")
        top_frame.pack(fill="x", pady=5)

        f_geo = ttk.LabelFrame(top_frame, text=" 📐 Géométrie ")
        f_geo.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ttk.Label(f_geo, text="Hauteur H (m) :", font=font_label).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.e_H = ttk.Entry(f_geo, width=10); self.e_H.insert(0, "10"); self.e_H.grid(row=0, column=1, padx=5)
        
        ttk.Label(f_geo, text="Largeur L (m) :", font=font_label).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="e")
        self.e_L = ttk.Entry(f_geo, width=10); self.e_L.insert(0, "20"); self.e_L.grid(row=1, column=1, padx=5, pady=(0, 10))

        f_cond = ttk.LabelFrame(top_frame, text=" ⚡ Sollicitations & Eau ")
        f_cond.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ttk.Label(f_cond, text="Niveau Nappe (m) :", font=font_label).grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.e_nap = ttk.Entry(f_cond, width=10); self.e_nap.insert(0, "-5"); self.e_nap.grid(row=0, column=1, padx=5)
        
        ttk.Label(f_cond, text="Surcharge q (kPa) :", font=font_label).grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.e_q = ttk.Entry(f_cond, width=10); self.e_q.insert(0, "0.0"); self.e_q.grid(row=1, column=1, padx=5)
        
        ttk.Label(f_cond, text="Séisme kh (fraction g) :", font=font_label).grid(row=2, column=0, padx=10, pady=(5, 10), sticky="e")
        self.e_kh = ttk.Entry(f_cond, width=10); self.e_kh.insert(0, "0.0"); self.e_kh.grid(row=2, column=1, padx=5, pady=(5, 10))

        # 2. STRATIGRAPHIE
        f_strat = ttk.LabelFrame(main_frame, text=" 🌍 Stratigraphie (Mettre Épaisseur = 0 pour ignorer) ")
        f_strat.pack(fill="x", pady=10)
        
        headers = ["Épaisseur (m)", "Cohésion c' (kPa)", "Frottement phi (°)", "Poids gamma (kN/m³)"]
        for j, h in enumerate(headers): 
            ttk.Label(f_strat, text=h, font=("Segoe UI", 9, "bold")).grid(row=0, column=j+1, pady=5, padx=10)
        
        self.couches_entries = []
        default_vals = [["6.0", "25.0", "25.0", "18.0"], ["10.0", "50.0", "30.0", "19.0"], ["0.0", "0.0", "0.0", "0.0"], ["0.0", "0.0", "0.0", "0.0"]]
        
        for i in range(4):
            ttk.Label(f_strat, text=f"Couche {i+1} :", font=font_label).grid(row=i+1, column=0, padx=10)
            row_entries = []
            for j in range(4):
                e = ttk.Entry(f_strat, width=12, justify="center")
                e.insert(0, default_vals[i][j])
                e.grid(row=i+1, column=j+1, padx=5, pady=4)
                row_entries.append(e)
            self.couches_entries.append(row_entries)

        # 3. BOUTONS D'ACTION (Design plat et coloré)
        f_btn = tk.Frame(main_frame, bg="#eceff1")
        f_btn.pack(pady=15)
        
        btn_strati = tk.Button(f_btn, text="👁️ Afficher la Stratigraphie", command=self.action_stratigraphie, 
                               bg="#2980b9", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", px=15, py=8, cursor="hand2")
        btn_strati.grid(row=0, column=0, padx=15)
        
        btn_analyse = tk.Button(f_btn, text="🚀 Lancer l'Analyse", command=self.action_analyse, 
                                bg="#27ae60", fg="white", font=("Segoe UI", 11, "bold"), relief="flat", px=15, py=8, cursor="hand2")
        btn_analyse.grid(row=0, column=1, padx=15)

    def valider_donnees(self):
        try:
            H, L = float(self.e_H.get()), float(self.e_L.get())
            y_n, q, kh = float(self.e_nap.get()), float(self.e_q.get()), float(self.e_kh.get())
            
            if H <= 0 or L <= 0: raise ValueError("H et L doivent être strictement positifs.")
            if q < 0: raise ValueError("La surcharge (q) ne peut pas être négative.")
            if kh < 0: raise ValueError("Le coefficient sismique (kh) ne peut pas être négatif.")
            
            couches, y_act = [], H
            colors = ['#FAD7A0','#E59866','#D4AC0D','#A04000']
            for i, row in enumerate(self.couches_entries):
                ep = float(row[0].get())
                if ep > 0:
                    c, ph, gam = float(row[1].get()), float(row[2].get()), float(row[3].get())
                    if c < 0 or ph < 0 or gam <= 0: 
                        raise ValueError(f"Paramètres (c, phi, gamma) invalides ou négatifs pour la Couche {i+1}.")
                    couches.append(CoucheSol(f"C{i+1}", y_act, y_act-ep, c, ph, gam, colors[len(couches)%4]))
                    y_act -= ep
            
            if not couches: raise ValueError("Il faut au moins une couche avec une épaisseur > 0.")
            couches[-1].y_bot = -50 
            
            return H, L, y_n, q, kh, couches
        except ValueError as e:
            messagebox.showwarning("⚠️ Erreur de Saisie", str(e))
            return None

    def dessiner_base(self, ax, couches, H, L, y_n):
        for c in couches: ax.fill_between([-L, L*2], c.y_bot, c.y_top, color=c.color, alpha=0.3, label=c.nom)
        ax.add_patch(Polygon([(-L,0), (0,0), (L,H), (L*2,H), (L*2,H*10), (-L,H*10)], facecolor='white', edgecolor='black', zorder=5))
        ax.axhline(y_n, color='#2980b9', lw=2, ls='--', label='Niveau de la Nappe')
        ax.set_aspect('equal'); ax.set_xlim(-L*0.5, L*1.5); ax.set_ylim(-15, H*2)

    def action_stratigraphie(self):
        donnees = self.valider_donnees()
        if not donnees: return
        H, L, y_n, q, kh, couches = donnees
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.dessiner_base(ax, couches, H, L, y_n)
        plt.title(f"Visualisation de la Stratigraphie\nPente: {math.degrees(math.atan(H/L)):.1f}°", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.show() 

    def action_analyse(self):
        donnees = self.valider_donnees()
        if not donnees: return
        H, L, y_n, q, kh, couches = donnees

        # Affichage d'un curseur d'attente
        self.root.config(cursor="wait")
        self.root.update()

        try:
            bounds = [(-L*0.5, L*1.5), (H*1.1, H*4), (H*0.5, H*5)]
            best_init_fs, x0 = 10.0, [L/2, H*2, H*1.5]
            for xc_t in np.linspace(0, L, 5):
                for yc_t in np.linspace(H*1.2, H*2.5, 5):
                    r_t = math.sqrt((xc_t - L/2)**2 + (yc_t - 0)**2)
                    f_t = calcul_bishop([xc_t, yc_t, r_t], couches, H, L, y_n, q, kh)
                    if f_t < best_init_fs: best_init_fs, x0 = f_t, [xc_t, yc_t, r_t]

            res = minimize(calcul_bishop, x0, args=(couches, H, L, y_n, q, kh, False), bounds=bounds, method='L-BFGS-B', tol=1e-4)
            fs_opt = res.fun
            xc_opt, yc_opt, R_opt = res.x

            grid_x, grid_y, grid_fs = [], [], []
            for x in np.linspace(xc_opt-5, xc_opt+5, 12):
                for y in np.linspace(yc_opt-5, yc_opt+5, 12):
                    f = calcul_bishop([x, y, R_opt], couches, H, L, y_n, q, kh, False)
                    if f < fs_opt * 2: grid_x.append(x); grid_y.append(y); grid_fs.append(f)

            _, slices = calcul_bishop([xc_opt, yc_opt, R_opt], couches, H, L, y_n, q, kh, True)

            fig, ax = plt.subplots(figsize=(12, 8))
            self.dessiner_base(ax, couches, H, L, y_n)
            
            norm_heatmap = plt.Normalize(vmin=fs_opt, vmax=fs_opt*1.2)
            sc = ax.scatter(grid_x, grid_y, c=grid_fs, cmap='viridis_r', norm=norm_heatmap, s=30, alpha=0.8, zorder=8, edgecolors='none')
            
            cmap_grad = plt.get_cmap('RdYlGn')
            for s in slices:
                rel_stab = s['ratio'] / fs_opt
                ax.add_patch(Polygon(s['coords'], facecolor=cmap_grad(plt.Normalize(0.8, 1.2)(rel_stab)), edgecolor='black', lw=0.4, zorder=6))

            t = np.linspace(0, 2*np.pi, 200)
            ax.plot(xc_opt + R_opt*np.cos(t), yc_opt + R_opt*np.sin(t), color='#e74c3c', lw=3.5, zorder=10, label=f'Rupture Critique (Fs={fs_opt:.3f})')
            
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(f"Facteur de Sécurité (Optimal: {fs_opt:.2f})", fontsize=10, fontweight="bold")
            
            plt.title(f"ANALYSE DE STABILITÉ - MÉTHODE DE BISHOP\nFacteur de Sécurité Global : {fs_opt:.3f}", fontweight="bold", fontsize=12)
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.show()

        finally:
            # Remettre le curseur normal une fois le calcul fini
            self.root.config(cursor="")

if __name__ == "__main__":
    root = tk.Tk()
    app = BishopApp(root)
    root.mainloop()