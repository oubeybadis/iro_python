# moteur_reactif_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ===== Interface =====
st.title("Simulation thermodynamique d'un moteur à réaction")
st.markdown("Calculs basés sur les lois fournies (relations isentropiques, atmosphère standard, bilans compresseur/combustion/turbine).")

st.sidebar.header("Paramètres environnementaux")
h = st.sidebar.number_input("Altitude de vol h [m]", value=0, step=100)  # altitude
# Atmosphère standard (troposphère)
T0 = 288.15 - 6.5 * h
# Pour la pression standard on utilise la formule de l'atmosphère standard (approximation troposphère)
P0 = 101325.0 * (1 - 6.5 * h / 288.15) ** 5.2559

st.sidebar.markdown("Valeurs initiales calculées")
st.sidebar.write(f"T0 = {T0:.2f} K")
st.sidebar.write(f"P0 = {P0:.1f} Pa")

st.sidebar.header("Paramètres moteur (modifiables)")
M0 = st.sidebar.number_input("Mach d'entrée M0", value=3.0, format="%.3f")
gamma = st.sidebar.number_input("γ (gamma)", value=1.4, format="%.3f")
R = st.sidebar.number_input("Constante gaz R [J/kg.K]", value=287.0)
Cp = st.sidebar.number_input("Cp [J/kg.K]", value=1005.0)
HPR = st.sidebar.number_input("HPR (J/kg) - pouvoir calorifique du carburant", value=42800e3, format="%.0f")

st.sidebar.markdown("Rapports et rendements")
pi_c = st.sidebar.number_input("Rapport de compression πc", value=12.0, format="%.3f")
eta_c = st.sidebar.number_input("Rendement compresseur ηc", value=0.85, format="%.3f")
eta_t = st.sidebar.number_input("Rendement turbine ηt", value=0.90, format="%.3f")
eta_b = st.sidebar.number_input("Rendement chambre de combustion ηb", value=0.95, format="%.3f")
pi_b_loss = st.sidebar.number_input("Rapport de perte de pression dans combustion (π_b_loss)", value=0.95, format="%.3f")

# ===== Calculs : lois issues des images =====
# 1) Conditions d'entrée (station 0)
# relations isentropiques pour transformer statique -> total
Tt0 = T0 * (1 + (gamma - 1) / 2 * M0 ** 2)  # T_t = T (1 + (γ-1)/2 M^2)
Pt0 = P0 * (1 + (gamma - 1) / 2 * M0 ** 2) ** (gamma / (gamma - 1))  # P_t = P (...)^(γ/(γ-1))

# On définit les stations principales :
# Station 0 : Entrée (conditions ambiantes ajustées à h)
# Station 1 : Après admission (on prend Tt0, Pt0 comme totals d'entrée)
# Station 2 : Après compresseur (Tt2, Pt2)
# Station 3 : Après chambre de combustion (Tt3, Pt3)
# Station 4 : Après turbine (Tt4, Pt4)
# Station 5 : Sortie tuyère (Tt5, Pt5) - on prend T5 ≈ T0 pour statique de sortie si besoin

# 2) Compresseur
# Loi utilisée (de la feuille) :
# Tt2/Tt1 = 1 + (1/ηc) * (πc^{(γ-1)/γ} - 1)
Tt1 = Tt0  # total entrée
Pt1 = Pt0
Tt2 = Tt1 * (1 + (1.0 / eta_c) * (pi_c ** ((gamma - 1) / gamma) - 1.0))
Pt2 = Pt1 * pi_c  # Pt2 = Pt1 * πc

# 3) Chambre de combustion
# On utilise l'équation de bilan énergétique donnée :
# f = Cp * (Tt3 - Tt2) / (ηb * HPR)
# On doit choisir une Tt3 cible (ex: température totale après combustion imposée).
# Dans les notes il y a souvent une valeur cible Tt3 (on peut la laisser comme paramètre ou estimer).
# Ici on propose un champ pour Tt3, sinon on prend une augmentation typique.
st.sidebar.header("Options chambre de combustion")
Tt3_input = st.sidebar.number_input("Température totale après combustion Tt3 [K] (0 = calcul automatique)", value=0.0)
if Tt3_input > 0:
    Tt3 = Tt3_input
else:
    # si non fourni, on augmente Tt2 d'une valeur indicative basée sur les notes (exemple +600 K)
    Tt3 = Tt2 + 600.0

# calcul du ratio carburant/air f
f = (Cp * (Tt3 - Tt2)) / (eta_b * HPR)

# perte de pression dans la chambre de combustion (π_b_loss)
Pt3 = Pt2 * pi_b_loss

# 4) Turbine
# Relation donnée dans la feuille (approx) :
# Tt4 = Tt3 - (Tt2 - Tt1) / ηt
# (l'idée : la turbine fournit le travail nécessaire pour le compresseur)
Tt4 = Tt3 - (Tt2 - Tt1) / eta_t

# Pour Pt4 on applique une relation isentropique approchée :
# Pt4 = Pt3 * (Tt4/Tt3)^(γ/(γ-1))  (approx. si on suppose une détente quasi-isentropique)
Pt4 = Pt3 * (Tt4 / Tt3) ** (gamma / (gamma - 1))

# 5) Tuyère (nozzle)
# On prend Tt5 = Tt4, Pt5 = Pt4 ; pour la statique de sortie on suppose P5 ≈ pression ambiante P0,
# ce qui permet de calculer le Mach de sortie et la vitesse si besoin.
Tt5 = Tt4
Pt5 = Pt4
P5 = P0  # approximation sortie vers ambiant

# Calculs des quantités demandées par station
stations = ["Entrée", "Compresseur", "Combustion", "Turbine", "Tuyère"]
# Pour l'affichage on donnera T_static et P_static sur chaque station:
# - station Entrée : T = T0, P = P0
# - pour les autres, on montrera T_total (Tt) et P_total (Pt) et les ratios Tt/T et Pt/P
T_static = [T0, None, None, None, None]
P_static = [P0, None, None, None, None]
Tt_list = [Tt0, Tt2, Tt3, Tt4, Tt5]
Pt_list = [Pt0, Pt2, Pt3, Pt4, Pt5]

# Pour les ratios Tt/T et Pt/P, on doit choisir le T et P statique de référence pour chaque station.
# On utilisera les valeurs statiques suivantes approximatives :
# - Après compresseur et après turbine la valeur statique peut être approximée par Tt / (1 + (γ-1)/2 * M^2)
#   mais sans M local on affiche les ratios Tt/T_ref où T_ref on le prend :
#   - pour compresseur : T_ref = Tt0 (approx total précédent considéré comme statique de référence)
#   - simplification : on présente Tt/T en utilisant T_statique proche disponible (Entrée pour station 1, Tt0 for next)
# Ici on calcule les ratios de la façon cohérente avec les formules de la fiche :
Tt_over_T = []
Pt_over_P = []

# station Entrée : ratio par rapport à T0 et P0
Tt_over_T.append(Tt0 / T0)
Pt_over_P.append(Pt0 / P0)

# station Compresseur : ratio par rapport à Tt1 (on suit la logique feuille)
Tt_over_T.append(Tt2 / Tt1)
Pt_over_P.append(Pt2 / Pt1)

# station Combustion : ratio par rapport à Tt2, Pt2
Tt_over_T.append(Tt3 / Tt2)
Pt_over_P.append(Pt3 / Pt2)

# station Turbine : ratio par rapport à Tt3, Pt3
Tt_over_T.append(Tt4 / Tt3)
Pt_over_P.append(Pt4 / Pt3)

# station Tuyère : ratio par rapport à Tt4, Pt4 (T statique de sortie approximée par T0)
Tt_over_T.append(Tt5 / Tt4 if Tt4 != 0 else np.nan)
Pt_over_P.append(Pt5 / P5 if P5 != 0 else np.nan)

# Tableau résultats
df = pd.DataFrame({
    "Station": stations,
    "T_total (K)": np.round(Tt_list, 2),
    "P_total (Pa)": np.round(Pt_list, 2),
    "Tt/T (rapport)": np.round(Tt_over_T, 4),
    "Pt/P (rapport)": np.round(Pt_over_P, 4)
})

st.subheader("Résultats numériques")
st.dataframe(df, use_container_width=True)

st.subheader("Paramètres calculés")
st.write(f"Température totale d'entrée Tt0 = {Tt0:.2f} K")
st.write(f"Pression totale d'entrée Pt0 = {Pt0:.1f} Pa")
st.write(f"Rapport carburant/air f = {f:.6f}")
st.write(f"Tt3 (après combustion) = {Tt3:.2f} K")
st.write(f"Tt4 (après turbine) = {Tt4:.2f} K")

# Graphique : Tt/T et Pt/P par station
st.subheader("Graphique : évolution des rapports Tt/T et Pt/P")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(stations, Tt_over_T, marker='o', label='Tt/T')
ax.plot(stations, Pt_over_P, marker='s', label='Pt/P')
ax.set_xlabel("Station")
ax.set_ylabel("Rapport")
ax.set_title("Tt/T et Pt/P selon les stations (lois de la feuille)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Export CSV
st.subheader("Téléchargements")
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button("Télécharger CSV", data=csv_data, file_name="resultats_moteur.csv", mime="text/csv")

# Export PDF
def generate_pdf(dataframe, params_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, height - 60, "Résultats de la simulation du moteur à réaction")
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 80, params_text)
    y = height - 110
    # Header
    cols = list(dataframe.columns)
    col_text = "  ".join([f"{cname}" for cname in cols])
    c.drawString(60, y, col_text)
    y -= 14
    # Rows
    for _, row in dataframe.iterrows():
        row_text = "  ".join([str(val) for val in row.values])
        c.drawString(60, y, row_text)
        y -= 12
        if y < 60:
            c.showPage()
            y = height - 60
    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

params_text = f"Altitude h={h} m, M0={M0}, πc={pi_c}, ηc={eta_c}, ηt={eta_t}, ηb={eta_b}"
pdf_data = generate_pdf(df, params_text)
st.download_button("Télécharger PDF", data=pdf_data, file_name="resultats_moteur.pdf", mime="application/pdf")

# Fin du script
