import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
# from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ============================
# Titre principal
# ============================
st.title("Simulation thermodynamique d’un moteur à réaction")
st.markdown("""
Ce programme calcule les températures et pressions dans les différentes stations d’un moteur à réaction,  
et trace les rapports **Tt/T** et **Pt/P** à travers les étapes principales.
""")

# ============================
# Paramètres d’entrée
# ============================
st.sidebar.header("Paramètres d’entrée")

M0 = st.sidebar.number_input("Nombre de Mach à l’entrée (M0)", value=3.0)
T0 = st.sidebar.number_input("Température ambiante T0 [K]", value=288.0)
P0 = st.sidebar.number_input("Pression ambiante P0 [Pa]", value=101325.0)
gamma = st.sidebar.number_input("Coefficient γ", value=1.4)
Cp = st.sidebar.number_input("Capacité calorifique Cp [J/kg.K]", value=1005.0)
R = st.sidebar.number_input("Constante des gaz R [J/kg.K]", value=287.0)
HPR = st.sidebar.number_input("Pouvoir calorifique du carburant HPR [J/kg]", value=42800e3)
pi_c = st.sidebar.number_input("Rapport de compression πc", value=12.0)
eta_c = st.sidebar.number_input("Rendement du compresseur ηc", value=0.85)
eta_t = st.sidebar.number_input("Rendement de la turbine ηt", value=0.9)
eta_b = st.sidebar.number_input("Rendement de la chambre de combustion ηb", value=0.95)

# ============================
# Calculs
# ============================

# Station 0 - Entrée
Tt0 = T0 * (1 + (gamma - 1)/2 * M0**2)
Pt0 = P0 * (1 + (gamma - 1)/2 * M0**2)**(gamma/(gamma - 1))

# Station 2 - Compresseur
Tt2 = Tt0 * (1 + (1/eta_c) * (pi_c**((gamma-1)/gamma) - 1))
Pt2 = Pt0 * pi_c

# Station 3 - Combustion
Tt3 = Tt2 + 600  # ajout de chaleur dans la chambre de combustion
f = (Cp * (Tt3 - Tt2)) / (eta_b * HPR)
Pt3 = Pt2 * 0.95  # légère perte de pression

# Station 4 - Turbine
Tt4 = Tt3 - (Tt2 - Tt0) / eta_t
Pt4 = Pt3 * (Tt4 / Tt3)**(gamma/(gamma-1))

# Station 5 - Tuyère
T5 = T0
Pt5 = Pt4
Tt5 = Tt4
P5 = P0

# ============================
# Résultats sous forme de tableau
# ============================
stations = ["Entrée", "Compresseur", "Combustion", "Turbine", "Tuyère"]
T_list = [T0, Tt0, Tt2, Tt3, Tt4]
P_list = [P0, Pt0, Pt2, Pt3, Pt4]
Tt_over_T = np.array([Tt0/T0, Tt2/Tt0, Tt3/Tt2, Tt4/Tt3, Tt5/T5])
Pt_over_P = np.array([Pt0/P0, Pt2/Pt0, Pt3/Pt2, Pt4/Pt3, Pt5/P5])

df = pd.DataFrame({
    "Station": stations,
    "Température (K)": np.round(T_list, 2),
    "Pression (Pa)": np.round(P_list, 2),
    "Tt/T": np.round(Tt_over_T, 4),
    "Pt/P": np.round(Pt_over_P, 4)
})

st.subheader("Résultats numériques")
st.dataframe(df, use_container_width=True)

# ============================
# Graphique
# ============================
st.subheader("Évolution des rapports Tt/T et Pt/P")

fig, ax = plt.subplots()
ax.plot(stations, Tt_over_T, marker='o', label='Tt/T', linewidth=2)
ax.plot(stations, Pt_over_P, marker='s', label='Pt/P', linewidth=2)
ax.set_xlabel("Stations du moteur")
ax.set_ylabel("Rapports")
ax.set_title("Variation des rapports Tt/T et Pt/P à travers le moteur")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ============================
# Informations supplémentaires
# ============================
st.subheader("Informations complémentaires")
st.write(f"Rapport carburant/air (f) = {f:.6f}")
st.write(f"Température après combustion (Tt3) = {Tt3:.2f} K")
st.write(f"Température en sortie de turbine (Tt4) = {Tt4:.2f} K")

csv = df.to_csv(index=False, sep=';', float_format='%.3f', encoding='utf-8')
st.download_button(
    "Télécharger les résultats (CSV)",
    csv,
    "resultats_moteur.csv",
    "text/csv"
)

# --- Generate clean PDF export ---
def generate_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Rapport de simulation du moteur à réaction", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Prepare table data
    data = [list(df.columns)] + df.values.tolist()
    formatted_data = []
    for row in data:
        formatted_row = [f"{x:.3f}" if isinstance(x, (float, int)) else str(x) for x in row]
        formatted_data.append(formatted_row)

    # Table styling
    table = Table(formatted_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightyellow]),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf_buffer = generate_pdf(df)
st.download_button(
    "Télécharger le rapport (PDF)",
    pdf_buffer,
    "rapport_moteur.pdf",
    "application/pdf"
)
