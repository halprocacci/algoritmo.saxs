import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, peak_prominences
from scipy.stats import linregress
import collections
import io
import datetime

st.set_page_config(page_title="SAXS Analyzer", layout="wide")
st.title("🔬 SAXS Lamellar Phase Analyzer")

# ==========================================
# ISTRUZIONI D'USO / INSTRUCTIONS FOR USE
# ==========================================
with st.expander("📖 Instructions for Use"):
    st.write(""" ciao
    *(Insert your operational instructions, formatting guidelines for the .csv, and theoretical background here...)*
    """)

# ==========================================
# SIDEBAR: PARAMETRI E OPZIONI UTENTE
# ==========================================
st.sidebar.header("⚙️ Processing Parameters")

# 1. Moltiplicatore Diamond (Ora con due bottoni)
st.sidebar.subheader("q-Axis Scaling")
if "multiply_q" not in st.session_state:
    st.session_state.multiply_q = False

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Input q in nm⁻¹"):
    st.session_state.multiply_q = False
if col_btn2.button("Input q in Å⁻¹"):
    st.session_state.multiply_q = True

applica_diamond = st.session_state.multiply_q
st.sidebar.caption(f"Current scaling state: **{'q in Å⁻¹' if applica_diamond else 'q in nm⁻¹'}**")


# 2. Window Length con preset guidati
preset_finestra = st.sidebar.selectbox(
    "Preset Window Length (Savitzky-Golay)",
    ["Custom", "Equilibrated Samples (29)", "Temperature Ramps (45)", "Diamond (11)"]
)
if preset_finestra == "Equilibrated Samples (29)":
    default_win = 29
elif preset_finestra == "Temperature Ramps (45)":
    default_win = 45
elif preset_finestra == "Diamond (11)":
    default_win = 11
else:
    default_win = 15

window_length = st.sidebar.slider("Window Length (must be an odd integer)", min_value=5, max_value=101, value=default_win, step=2)
polyorder = st.sidebar.slider("Polynomial Order (Polyorder)", min_value=1, max_value=5, value=3)

# 3. Taglio Range q
st.sidebar.subheader("q-Axis Range")
col_q1, col_q2 = st.sidebar.columns(2)
q_min = col_q1.number_input("q min", value=0.4, step=0.1)
q_max = col_q2.number_input("q max", value=5.2, step=0.1)

# 4. Parametri Regressione e Riconoscimento Fasi
st.sidebar.subheader("Crystallographic Thresholds")
prominenza_min = st.sidebar.number_input("Minimum Peak Prominence Threshold", value=0.0010, format="%.4f", step=0.0005)
r2_singolo_min = st.sidebar.number_input("Minimum R² Threshold (Single)", value=0.9997, format="%.4f", step=0.0001)

# ==========================================
# ERC FUNDING ACKNOWLEDGEMENT
# ==========================================
st.sidebar.markdown("---")
st.sidebar.caption("""
**Funding Acknowledgement:**  
grant agreement No. 949229, CryForm.
""")

# ==========================================
# FUNZIONI CORE (CALCOLI IN MEMORIA)
# ==========================================
def elabora_matrici(file_buffer, x_mult, win, poly, q_min, q_max):
    """Sostituisce analisi.2.py calcolando tutto in memoria RAM senza scrivere su disco."""
    x_values, y_data, labels = [], [], []
    
    file_str = io.StringIO(file_buffer.getvalue().decode("utf-8", errors="ignore"))
    for i, line in enumerate(file_str):
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if i == 0:
            x_values = [float(val.replace(",", ".")) for val in parts[1:] if val.strip() != ""]
        else:
            labels.append(parts[0])
            y_row = [float(val.replace(",", ".")) for val in parts[1:len(x_values)+1] if val.strip() != ""]
            y_data.append(y_row)

    x_values = np.array(x_values)
    y_data = np.array(y_data)

    if x_mult:
        x_values = x_values * 10.0

    # Taglio q
    indici = np.where((x_values >= q_min) & (x_values <= q_max))[0]
    x_values = x_values[indici]
    y_data = y_data[:, indici]

    dx = x_values[1] - x_values[0]
    y_filtered = np.zeros_like(y_data)
    derivate = np.zeros_like(y_data)

    for i in range(len(labels)):
        y_filtered[i] = savgol_filter(y_data[i], window_length=win, polyorder=poly)
        derivate[i] = savgol_filter(y_data[i], window_length=win, polyorder=poly, deriv=1, delta=dx)

    # Identificazione picchi + Prominenza topografica centrata (±4 punti)
    matrice_con_prominence = {}
    
    for idx in range(len(labels)):
        riga_deriv = derivate[idx]
        riga_y = y_filtered[idx]
        
        # Attraversamento dello zero (>0 a <0)
        indici_zerocross = []
        for j in range(len(riga_deriv) - 1):
            if riga_deriv[j] > 0 and riga_deriv[j+1] < 0:
                indici_zerocross.append(j)

        if not indici_zerocross:
            matrice_con_prominence[idx + 1] = []
            continue

        # Ricentratura picco sul massimo locale
        indici_corretti = []
        for idx_p in indici_zerocross:
            start = max(0, idx_p - 4)
            end = min(len(riga_y), idx_p + 5)
            idx_max = start + np.argmax(riga_y[start:end])
            indici_corretti.append(idx_max)

        indici_corretti = sorted(list(set(indici_corretti)))
        prominenze = peak_prominences(riga_y, indici_corretti)[0]

        picchi = []
        for p_i, idx_c in enumerate(indici_corretti):
            picchi.append({
                'q': round(x_values[idx_c], 4),
                'prom': prominenze[p_i]
            })

        # Ordina per prominenza, tieni i top 30, poi riordina per q crescente
        picchi.sort(key=lambda x: x['prom'], reverse=True)
        top_30 = picchi[:30]
        top_30.sort(key=lambda x: x['q'])

        matrice_con_prominence[idx + 1] = [(p['q'], p['prom']) for p in top_30]

    return x_values, y_data, y_filtered, derivate, labels, matrice_con_prominence

def calcola_regressione(picchi_riga, prom_min, r2_threshold):
    """Logica di clustering lamellare e fit R^2 derivata da rettacompleta.py."""
    serie = collections.defaultdict(list)
    lineasoloq = []
    doppio = True

    for el in picchi_riga:
        q_val, prom_val = el[0], el[1]
        if prom_val > prom_min and 0.5 < q_val < 1.5:
            lineasoloq.append(q_val)
            serie[q_val].append((0, 0))
            serie[q_val].append((1, q_val))
            if 0.7 < q_val < 1.0:
                doppio = False
        elif q_val > 1.5:
            lineasoloq.append(q_val)
            if doppio and 1.5 < q_val < 2.0 and prom_val > prom_min:
                serie[q_val].append((0, 0))
                serie[q_val].append((1, q_val))
            elif 1.5 < q_val < 2.0 and prom_val > prom_min:
                aggiungi = True
                for el2 in lineasoloq:
                    if (2 * el2 - 0.08) < q_val < (2 * el2 + 0.08):
                        aggiungi = False
                if aggiungi:
                    serie[q_val].append((0, 0))
                    serie[q_val].append((1, q_val))

    risultati_fasi = []
    
    for key in serie:
        listar = []
        punti_finali = serie[key].copy()
        m_fit = None

        for el in lineasoloq:
            rapporto = el / key
            round_r = round(rapporto)
            if round_r > 1 and abs(rapporto - round_r) < 0.2:
                candidata = punti_finali.copy()
                candidata.append((round_r, el))
                
                x = [p[0] for p in candidata]
                y = [p[1] for p in candidata]
                res = linregress(x, y)
                r2 = res.rvalue ** 2

                if r2 > 0.997 and punti_finali[-1][0] == round_r:
                    candidata_sost = punti_finali[:-1] + [(round_r, el)]
                    x2 = [p[0] for p in candidata_sost]
                    y2 = [p[1] for p in candidata_sost]
                    res2 = linregress(x2, y2)
                    r2_2 = res2.rvalue ** 2
                    if listar and r2_2 > listar[-1]:
                        punti_finali = candidata_sost
                        listar[-1] = r2_2
                        m_fit = res2.slope
                elif r2 > r2_threshold:
                    listar.append(r2)
                    punti_finali = candidata
                    m_fit = res.slope

        if m_fit is not None:
            d_spacing = (2 * np.pi) / m_fit
            risultati_fasi.append({
                'q1': key,
                'punti': punti_finali,
                'R2': listar,
                'slope': m_fit,
                'd_spacing': d_spacing
            })

    return risultati_fasi

# ==========================================
# INTERFACCIA PRINCIPALE
# ==========================================
uploaded_file = st.file_uploader("Upload SAXS Data File (semicolon-delimited .csv)", type=["csv", "txt"])

if uploaded_file is not None:
    # Esegue il filtraggio e l'elaborazione del segnale
    with st.spinner("Filtering and computing signal derivatives..."):
        x_vals, y_raw, y_filt, deriv, labels, picchi_dict = elabora_matrici(
            uploaded_file, applica_diamond, window_length, polyorder, q_min, q_max
        )

    st.success(f"Data successfully loaded: {len(labels)} samples/temperatures identified.")

    # TABS INTERFACCIA
    tab_grafici, tab_singola, tab_batch = st.tabs(["📊 Signal Plots", "🎯 Single-Spectrum Analysis", "🚀 Batch Analysis"])

    # ---------------- TAB 1: GRAFICI ----------------
    with tab_grafici:
        st.subheader("Spectra and Derivative Diagnostics")
        
        # Scelta tra vista singola o sovrapposizione globale
        modalita_grafico = st.radio(
            "Display Mode:",
            ["Overlay All Spectra", "Single Spectrum"],
            horizontal=True
        )

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

        if modalita_grafico == "Overlay All Spectra":
            # Mostra tutte le righe insieme
            for i in range(len(labels)):
                ax1.plot(x_vals, y_raw[i], label=labels[i], alpha=0.7)
                ax2.plot(x_vals, y_filt[i], label=labels[i], alpha=0.7)
                ax3.plot(x_vals, deriv[i], label=labels[i], alpha=0.7)

            ax1.set_title("1. Raw Scattering Curves")
            ax2.set_title(f"2. Filtered Curves (win={window_length})")
            ax3.set_title("3. Exact First Derivatives")
            
            # Mostra la legenda solo se non ci sono troppi campioni da coprire il grafico
            if len(labels) <= 15:
                ax3.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')

        else:
            # Vista singola riga
            idx_grafico = st.selectbox(
                "Select Sample for Visualization",
                options=range(len(labels)),
                format_func=lambda i: f"Row {i+1}: {labels[i]}"
            )
            ax1.plot(x_vals, y_raw[idx_grafico], color='black', label="Raw")
            ax1.set_title(f"1. Raw Spectrum ({labels[idx_grafico]})")

            ax2.plot(x_vals, y_filt[idx_grafico], color='blue', label=f"S-G (win={window_length})")
            ax2.set_title("2. Filtered Curve")

            ax3.plot(x_vals, deriv[idx_grafico], color='red', label="dy/dx")
            ax3.set_title("3. Exact First Derivative")

        # Impostazioni assi comuni
        ax1.set_yscale('log')
        ax1.set_xlabel("q (Å⁻¹)")
        ax1.grid(True, alpha=0.3)

        ax2.set_yscale('log')
        ax2.set_xlabel("q (Å⁻¹)")
        ax2.grid(True, alpha=0.3)

        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel("q (Å⁻¹)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        
    # ---------------- TAB 2: SINGOLA RIGA ----------------
    with tab_singola:
        st.subheader("Linear Regression and d-Spacing Calculation")
        riga_scelta = st.number_input("Select Row Index for Analysis", min_value=1, max_value=len(labels), value=1)
        
        if st.button("Execute Regression for Selected Row"):
            picchi_target = picchi_dict.get(riga_scelta, [])
            risultati = calcola_regressione(picchi_target, prominenza_min, r2_singolo_min)
            
            st.write(f"**Sample:** {labels[riga_scelta - 1]}")
            if not risultati:
                st.warning("No lamellar phase aligns with the defined crystallographic criteria.")
            else:
                for idx_fase, res in enumerate(risultati):
                    st.markdown(f"### Identified Phase #{idx_fase + 1} (Fundamental Peak at q ≈ {res['q1']})")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(label="Calculated d-spacing", value=f"{res['d_spacing']:.4f} Å")
                        st.write("**Identified Diffraction Orders (n, q):**")
                        st.write(res['punti'][1:])
                        st.write(f"**Sequential R² Values:** {res['R2']}")
                    
                    with col2:
                        fig_fit, ax_fit = plt.subplots(figsize=(6, 3.5))
                        x_pts = [p[0] for p in res['punti']]
                        y_pts = [p[1] for p in res['punti']]
                        ax_fit.scatter(x_pts, y_pts, color='crimson', zorder=3, label="Experimental Peaks")
                        x_line = np.linspace(0, max(x_pts), 50)
                        ax_fit.plot(x_line, res['slope'] * x_line, '--', color='navy', label=f"Fit (m={res['slope']:.4f})")
                        ax_fit.set_xlabel("Diffraction Order (n)")
                        ax_fit.set_ylabel("q (Å⁻¹)")
                        ax_fit.grid(True, alpha=0.3)
                        ax_fit.legend()
                        st.pyplot(fig_fit)

    # ---------------- TAB 3: BATCH ANALYSIS ----------------
    with tab_batch:
        st.subheader("Automated Batch Processing")
        tipo_batch = st.radio("Select rows for processing:", ["All rows in dataset", "Custom range"])
        
        if tipo_batch == "Custom range":
            col_b1, col_b2 = st.columns(2)
            b_start = col_b1.number_input("Start row", min_value=1, max_value=len(labels), value=1)
            b_end = col_b2.number_input("End row", min_value=1, max_value=len(labels), value=len(labels))
            righe_batch = list(range(b_start, b_end + 1))
        else:
            righe_batch = list(range(1, len(labels) + 1))

        if st.button("Run Batch Analysis"):
            log_batch = f"--- SAXS ANALYSIS LOG | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ---\n\n"
            progress_bar = st.progress(0)
            
            dati_tabella = []
            for count, r_idx in enumerate(righe_batch):
                picchi_target = picchi_dict.get(r_idx, [])
                risultati = calcola_regressione(picchi_target, prominenza_min, r2_singolo_min)
                
                label_campione = labels[r_idx - 1]
                log_batch += f"Row {r_idx} | {label_campione}\n"
                
                if not risultati:
                    log_batch += "  No lamellar phase detected.\n"
                else:
                    for res in risultati:
                        log_batch += f"  Phase (q1={res['q1']}): {res['punti']}\n"
                        log_batch += f"  R^2: {res['R2']}\n"
                        log_batch += f"  d-spacing: {res['d_spacing']:.4f} Å\n"
                        
                        dati_tabella.append({
                            "Row": r_idx,
                            "Sample": label_campione,
                            "q1 (Å⁻¹)": res['q1'],
                            "d-spacing (Å)": round(res['d_spacing'], 4),
                            "Final R²": round(res['R2'][-1], 6) if res['R2'] else None,
                            "Peak Count": len(res['punti']) - 1
                        })
                log_batch += "---------------------------------------------------\n"
                progress_bar.progress((count + 1) / len(righe_batch))

            st.success("Batch Analysis Completed Successfully!")
            
            if dati_tabella:
                st.dataframe(pd.DataFrame(dati_tabella), use_container_width=True)
            
            st.download_button(
                label="📥 Download Analysis Log (.txt)",
                data=log_batch,
                file_name="batch_analysis_log.txt",
                mime="text/plain"
            )
