import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, shapiro
import scikit_posthocs as sp

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SALiNA — Dashboard de Evaluación",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos globales ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1565C0;
        margin-bottom: 0.5rem;
    }
    h1 { color: #1a237e; }
    h2 { color: #283593; border-bottom: 2px solid #e8eaf6; padding-bottom: 4px; }
    h3 { color: #3949ab; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
WEB   = 'salina.web.ua.pt'
AI4   = 'salina.ai4life.uk'
LANGS = ['Português', 'English', 'Español']
CATS_CUANT = ['Búsqueda de libros', 'Búsqueda por autor']
CATS_CUAL  = ['Artículos científicos', 'Revistas y publicaciones', 'Metadatos del catálogo']
ALPHA = 0.05

SYS_PAL  = {WEB: '#1565C0', AI4: '#C62828'}
LANG_PAL = {'Português': '#4472C4', 'English': '#70AD47', 'Español': '#ED7D31'}
CAT_PAL  = {
    'Búsqueda de libros'       : '#7B1FA2',
    'Búsqueda por autor'       : '#00796B',
    'Artículos científicos'    : '#E65100',
    'Revistas y publicaciones' : '#0277BD',
    'Metadatos del catálogo'   : '#558B2F',
}

COL_EXT_WEB = 'Respuesta extraida del chatbot https://salina.web.ua.pt/'
COL_EXT_AI4 = 'Respuesta extraida del chatbot https://salina.ai4life.uk/'
COL_RESP_WEB = 'Respuesta del chatbot https://salina.web.ua.pt/'
COL_RESP_AI4 = 'Respuesta del chatbot https://salina.ai4life.uk/'
COL_MAN      = 'Respuesta busqueda manual'
COL_Q        = 'Texto de la pregunta'

plt.rcParams.update({
    'figure.dpi'        : 120,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'font.size'         : 9,
})
sns.set_theme(style='whitegrid', font_scale=0.95)

# ── Funciones estadísticas ────────────────────────────────────────────────────
def epsilon_squared(H, k, n):
    return (H - k + 1) / (n - k)

def rank_biserial_r(U, n1, n2):
    return 1 - (2 * U / (n1 * n2))

def eff_eps(v):
    return 'grande' if abs(v) >= 0.16 else ('medio' if abs(v) >= 0.04 else 'pequeño')

def eff_r(v):
    return 'grande' if abs(v) >= 0.5 else ('medio' if abs(v) >= 0.3 else 'pequeño')

def sig_badge(p):
    if p < 0.001:
        return "🟢 p < 0.001 — Muy significativo"
    elif p < 0.01:
        return "🟢 p < 0.01 — Significativo"
    elif p < 0.05:
        return "🟡 p < 0.05 — Significativo"
    else:
        return "🔴 p ≥ 0.05 — No significativo"

# ── Carga y preparación de datos ─────────────────────────────────────────────
@st.cache_data
def cargar_datos(archivo):
    raw = pd.read_excel(archivo)

    # Sección cuantitativa
    dq = raw[raw['Categoría'].isin(CATS_CUANT)].copy().reset_index(drop=True)
    dq = dq.rename(columns={
        'Idioma': 'idioma', 'Categoría': 'categoria',
        COL_EXT_WEB: 'web', COL_EXT_AI4: 'ai4', COL_MAN: 'manual',
    })
    dq['ae_web']  = (dq['web'] - dq['manual']).abs()
    dq['ae_ai4']  = (dq['ai4'] - dq['manual']).abs()
    dq['err_web'] = dq['web'] - dq['manual']
    dq['err_ai4'] = dq['ai4'] - dq['manual']

    # Sección cualitativa
    dc = raw[raw['Categoría'].isin(CATS_CUAL)].copy().reset_index(drop=True)
    dc = dc.rename(columns={
        'Idioma': 'idioma', 'Categoría': 'categoria',
        COL_RESP_WEB: 'resp_web', COL_RESP_AI4: 'resp_ai4', COL_Q: 'pregunta',
    })
    NEG = [
        'não foi possível', 'não encontr', 'ocorreu um erro', 'lamentavelmente',
        'nenhum resultado', 'não há resultado', 'unable to', 'could not',
        'not found', 'no results', 'unfortunately', 'no encontr', 'no se pudo',
        'no hay resultado', 'lamentablemente', 'there are no', 'sem resultados',
    ]
    dc['neg_web'] = dc['resp_web'].apply(
        lambda t: any(p in str(t).lower() for p in NEG))
    dc['neg_ai4'] = dc['resp_ai4'].apply(
        lambda t: any(p in str(t).lower() for p in NEG))

    return dq, dc

# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        'https://i0.wp.com/cdcs.web.ua.pt/wp-content/uploads/2022/05/cropped-cropped-Picture13-1.png?w=968',
        width=220
    )
    st.markdown("### 📚 SALiNA Dashboard")
    st.markdown("*Evaluación multilingüe de rendimiento RAG*")
    st.divider()

    archivo = st.file_uploader("📁 Sube el archivo Excel", type=['xlsx'])

    if archivo:
        st.divider()
        st.markdown("### ⚙️ Filtros")
        seccion = st.radio(
            "Sección de análisis",
            ["🔢 Cuantitativo (MAE)", "📝 Cualitativo (LaBSE)", "📊 Resumen ejecutivo"],
            index=0
        )
        idiomas_sel = st.multiselect(
            "Idiomas", LANGS, default=LANGS
        )
        st.divider()
        st.caption(f"α = {ALPHA} | Tests no paramétricos")

# ════════════════════════════════════════════════════════════════════════════════
#  PANTALLA PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════
if not archivo:
    st.title("📚 SALiNA — Dashboard de Evaluación Multilingüe")
    st.markdown("""
    Este dashboard reproduce el análisis estadístico del **Milestone 3 — Bloque 1**,
    evaluando el rendimiento observable de las dos versiones del sistema RAG SALiNA
    ante consultas en tres idiomas.

    | Sistema | URL |
    |---|---|
    | **salina.web** | https://salina.web.ua.pt/ |
    | **salina.ai4life** | https://salina.ai4life.uk/ |

    ---
    👈 **Sube el archivo `Respuestas3.xlsx` en el panel lateral para comenzar.**
    """)
    st.stop()

# ── Carga de datos ────────────────────────────────────────────────────────────
dq, dc = cargar_datos(archivo)
langs_activos = [l for l in LANGS if l in idiomas_sel] or LANGS

# ════════════════════════════════════════════════════════════════════════════════
#  SECCIÓN CUANTITATIVA
# ════════════════════════════════════════════════════════════════════════════════
if seccion == "🔢 Cuantitativo (MAE)":

    st.title("🔢 Análisis Cuantitativo — Error Absoluto (MAE)")
    st.markdown(
        "Categorías: **Búsqueda de libros** y **Búsqueda por autor**. "
        "Métrica: Error Absoluto entre la respuesta del RAG y el valor real del catálogo."
    )

    dq_f = dq[dq['idioma'].isin(langs_activos)]

    # ── Métricas globales ─────────────────────────────────────────────────────
    st.subheader("Métricas globales")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE salina.web",    f"{dq_f['ae_web'].mean():.2f}")
    c2.metric("MAE salina.ai4life", f"{dq_f['ae_ai4'].mean():.2f}")
    c3.metric("Aciertos exactos web",
              f"{(dq_f['ae_web']==0).sum()}/{len(dq_f)}")
    c4.metric("Aciertos exactos ai4life",
              f"{(dq_f['ae_ai4']==0).sum()}/{len(dq_f)}")

    st.divider()

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Descriptivos",
        "🔵 Real vs RAG",
        "📐 Sesgo",
        "H1 — Kruskal-Wallis",
        "H2 — Mann-Whitney"
    ])

    # ── Tab 1: Descriptivos ───────────────────────────────────────────────────
    with tab1:
        st.subheader("Estadísticos descriptivos")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Por idioma (MAE medio)**")
            tbl = dq_f.groupby('idioma')[['ae_web','ae_ai4']].agg(
                ['mean','median']).round(2)
            tbl.columns = ['Media web','Mediana web','Media ai4','Mediana ai4']
            st.dataframe(tbl.loc[[l for l in LANGS if l in langs_activos]],
                         use_container_width=True)

        with col2:
            st.markdown("**Por categoría (MAE medio)**")
            tbl2 = dq_f.groupby('categoria')[['ae_web','ae_ai4']].agg(
                ['mean','median']).round(2)
            tbl2.columns = ['Media web','Mediana web','Media ai4','Mediana ai4']
            st.dataframe(tbl2, use_container_width=True)

        st.markdown("**Distribución del Error Absoluto**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (sn, col) in zip(axes, [(WEB,'ae_web'),(AI4,'ae_ai4')]):
            for lang in langs_activos:
                sub = dq_f[dq_f['idioma']==lang][col]
                ax.hist(sub, bins=12, alpha=0.6, label=lang,
                        color=LANG_PAL[lang], edgecolor='white')
            ax.axvline(dq_f[col].median(), color=SYS_PAL[sn],
                       linestyle='--', linewidth=1.8,
                       label=f'Mediana={dq_f[col].median():.1f}')
            ax.set_xlabel('Error Absoluto (AE)')
            ax.set_ylabel('Frecuencia')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            ax.legend(fontsize=8)
        fig.suptitle('Distribución del AE — asimetría justifica tests no paramétricos',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Real vs RAG ────────────────────────────────────────────────────
    with tab2:
        st.subheader("Valor real del catálogo vs respuesta del RAG")
        st.caption("Puntos sobre la diagonal = sobreestimación | Bajo = subestimación")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        max_val = max(dq_f[['manual','web','ai4']].max()) * 1.05

        for ax, (sn, col) in zip(axes, [(WEB,'web'),(AI4,'ai4')]):
            for lang in langs_activos:
                sub = dq_f[dq_f['idioma']==lang]
                ax.scatter(sub['manual'], sub[col],
                           color=LANG_PAL[lang], alpha=0.75, s=55,
                           label=lang, edgecolors='white', linewidth=0.5)
            ax.plot([0,max_val],[0,max_val],'k--',linewidth=1.2,
                    alpha=0.6, label='Respuesta perfecta')
            ax.set_xlabel('Valor real (manual)')
            ax.set_ylabel(f'Respuesta {sn}')
            ax.set_title(f'{sn}\nMAE={dq_f[col.replace("web","ae_web").replace("ai4","ae_ai4")].mean():.1f}' if False else sn,
                         fontweight='bold', color=SYS_PAL[sn])
            ax.set_xlim(-5, max_val); ax.set_ylim(-5, max_val)
            ax.legend(fontsize=8)

        # Fix titles with MAE
        axes[0].set_title(f'{WEB}\nMAE = {dq_f["ae_web"].mean():.1f}',
                          fontweight='bold', color=SYS_PAL[WEB])
        axes[1].set_title(f'{AI4}\nMAE = {dq_f["ae_ai4"].mean():.1f}',
                          fontweight='bold', color=SYS_PAL[AI4])

        fig.suptitle('Respuesta del RAG vs valor real del catálogo', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Barras agrupadas
        st.markdown("**Comparación pregunta a pregunta**")
        dq_sorted = dq_f.sort_values(['categoria','idioma']).reset_index(drop=True)
        n_show = min(30, len(dq_sorted))
        sub30  = dq_sorted.iloc[:n_show]
        x = np.arange(n_show); w = 0.28

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.bar(x-w, sub30['manual'], w, label='Valor real', color='#37474F', alpha=0.85)
        ax.bar(x,   sub30['web'],    w, label=WEB,           color=SYS_PAL[WEB], alpha=0.8)
        ax.bar(x+w, sub30['ai4'],    w, label=AI4,           color=SYS_PAL[AI4], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['idioma'][:3]}" for _, r in sub30.iterrows()],
            fontsize=7
        )
        ax.set_ylabel('Cantidad de resultados')
        ax.set_title(f'Valor real vs respuestas (primeros {n_show} registros)', fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: Sesgo ──────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Sesgo de sobre/subestimación")
        st.caption("Arriba de la línea cero = sobreestima | Abajo = subestima")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (sn, err_col) in zip(axes, [(WEB,'err_web'),(AI4,'err_ai4')]):
            for lang in langs_activos:
                sub = dq_f[dq_f['idioma']==lang]
                ax.scatter(sub['manual'], sub[err_col],
                           color=LANG_PAL[lang], alpha=0.75, s=55,
                           label=lang, edgecolors='white')
            ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
            sesgo = dq_f[err_col].mean()
            ax.axhline(sesgo, color=SYS_PAL[sn], linewidth=1.5,
                       linestyle=':', label=f'Sesgo medio = {sesgo:+.1f}')
            ax.set_xlabel('Valor real (manual)')
            ax.set_ylabel('Error con signo (respuesta − real)')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            ax.legend(fontsize=8)

        fig.suptitle('Sesgo de sobre/subestimación por sistema', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Heatmap MAE
        st.markdown("**Heatmap MAE por idioma y categoría**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        for ax, (sn, col) in zip(axes, [(WEB,'ae_web'),(AI4,'ae_ai4')]):
            pivot = dq_f.pivot_table(
                values=col, index='idioma', columns='categoria', aggfunc='mean'
            ).round(1)
            pivot = pivot.loc[[l for l in LANGS if l in pivot.index],
                               [c for c in CATS_CUANT if c in pivot.columns]]
            sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd',
                        linewidths=0.5, linecolor='#eee',
                        cbar_kws={'label': 'MAE (media)'})
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            ax.tick_params(axis='x', rotation=15)
        fig.suptitle('Heatmap — MAE por idioma y categoría', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 4: H1 Kruskal-Wallis ──────────────────────────────────────────────
    with tab4:
        st.subheader("H1 — ¿El idioma afecta el rendimiento observable del RAG?")
        st.markdown("""
        **Kruskal-Wallis** compara las distribuciones del Error Absoluto entre los tres idiomas.
        Un resultado significativo indica que al menos un idioma produce errores sistemáticamente distintos.
        """)

        for sn, col in [(WEB,'ae_web'),(AI4,'ae_ai4')]:
            grupos  = [dq_f[dq_f['idioma']==lang][col].values
                       for lang in langs_activos if lang in dq_f['idioma'].values]
            if len(grupos) < 2:
                continue
            n_total = sum(len(g) for g in grupos)
            k       = len(grupos)
            H, p    = kruskal(*grupos)
            eps2    = epsilon_squared(H, k, n_total)

            with st.expander(f"**{sn}** — H={H:.3f}  p={p:.4f}  ε²={eps2:.4f} ({eff_eps(eps2)})",
                             expanded=True):
                st.markdown(f"**Decisión:** {sig_badge(p)}")

                cols = st.columns(len(langs_activos))
                for col_ui, (lang, g) in zip(cols, zip(langs_activos, grupos)):
                    col_ui.metric(
                        lang,
                        f"Med={np.median(g):.1f}",
                        f"n={len(g)}"
                    )

                if p < ALPHA and len(grupos) >= 3:
                    st.markdown("**Post-hoc Dunn + Bonferroni:**")
                    long_df = pd.concat([
                        pd.DataFrame({'val': g, 'idioma': lang})
                        for g, lang in zip(grupos, langs_activos)
                    ], ignore_index=True)
                    dunn = sp.posthoc_dunn(long_df, val_col='val',
                                           group_col='idioma', p_adjust='bonferroni')
                    rows = []
                    for i, l1 in enumerate(langs_activos):
                        for l2 in langs_activos[i+1:]:
                            pv = dunn.loc[l1, l2]
                            rows.append({'Par': f'{l1} vs {l2}',
                                         'p ajustado': f'{pv:.4f}',
                                         'Significativo': '✓' if pv < ALPHA else '✗'})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Boxplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (sn, col) in zip(axes, [(WEB,'ae_web'),(AI4,'ae_ai4')]):
            data = [dq_f[dq_f['idioma']==lang][col].values
                    for lang in langs_activos]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color='black', linewidth=2))
            for patch, lang in zip(bp['boxes'], langs_activos):
                patch.set_facecolor(LANG_PAL[lang]); patch.set_alpha(0.78)
            for i, g in enumerate(data):
                ax.scatter(i+1, np.mean(g), color='white', s=55,
                           zorder=5, edgecolors='black', linewidth=1.2)
            H, p = kruskal(*data)
            eps2 = epsilon_squared(H, len(data), sum(len(g) for g in data))
            cb = '#2E7D32' if p < ALPHA else '#B71C1C'
            ax.text(0.5, 0.97, f'H={H:.2f}  p={p:.3f}  ε²={eps2:.3f}' +
                    ('  ✓' if p < ALPHA else '  ✗'),
                    transform=ax.transAxes, ha='center', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor=cb, boxstyle='round', alpha=0.9))
            ax.set_xticks(range(1, len(langs_activos)+1))
            ax.set_xticklabels(langs_activos)
            ax.set_ylabel('Error Absoluto (AE)')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])

        patches = [mpatches.Patch(color=LANG_PAL[l], label=l) for l in langs_activos]
        fig.legend(handles=patches, title='Idioma', loc='lower center',
                   ncol=len(langs_activos), bbox_to_anchor=(0.5, -0.04))
        fig.suptitle('H1 — AE por idioma (Kruskal-Wallis)\nCírculo blanco = media',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 5: H2 Mann-Whitney ────────────────────────────────────────────────
    with tab5:
        st.subheader("H2 — ¿salina.web y salina.ai4life difieren por idioma?")
        st.markdown("""
        **Mann-Whitney U** compara los dos sistemas dentro de cada idioma.
        Un resultado significativo indica que un sistema produce errores menores en ese idioma.
        """)

        rows = []
        for lang in langs_activos:
            sub = dq_f[dq_f['idioma']==lang]
            g_w = sub['ae_web'].values; g_a = sub['ae_ai4'].values
            U, p = mannwhitneyu(g_w, g_a, alternative='two-sided')
            r    = rank_biserial_r(U, len(g_w), len(g_a))
            rows.append({
                'Idioma': lang,
                'U': f'{U:.1f}',
                'p': f'{p:.4f}',
                'r': f'{r:.3f}',
                'Efecto': eff_r(r),
                'Sig.': '✓' if p < ALPHA else '✗',
                'Mejor (menor AE)': (WEB if np.median(g_w) < np.median(g_a) else AI4)
                                    if p < ALPHA else '—',
                'Med. web': f'{np.median(g_w):.1f}',
                'Med. ai4life': f'{np.median(g_a):.1f}',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Boxplots por idioma
        fig, axes = plt.subplots(1, len(langs_activos),
                                  figsize=(5*len(langs_activos), 5))
        if len(langs_activos) == 1:
            axes = [axes]
        for ax, lang in zip(axes, langs_activos):
            sub = dq_f[dq_f['idioma']==lang]
            g_w = sub['ae_web'].values; g_a = sub['ae_ai4'].values
            bp  = ax.boxplot([g_w, g_a], patch_artist=True,
                              medianprops=dict(color='black', linewidth=2.2))
            for patch, color in zip(bp['boxes'],
                                    [SYS_PAL[WEB], SYS_PAL[AI4]]):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            for i, g in enumerate([g_w, g_a]):
                ax.scatter(i+1, np.mean(g), color='white', s=55,
                           zorder=5, edgecolors='black', linewidth=1.2)
            U, p = mannwhitneyu(g_w, g_a, alternative='two-sided')
            r    = rank_biserial_r(U, len(g_w), len(g_a))
            cb   = '#2E7D32' if p < ALPHA else '#B71C1C'
            ax.text(0.5, 0.03, f'p={p:.3f}  r={r:.3f}' + ('  ✓' if p < ALPHA else '  ✗'),
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor=cb, boxstyle='round', alpha=0.9))
            ax.set_xticks([1,2])
            ax.set_xticklabels(['salina.web','salina.ai4life'], fontsize=8)
            ax.set_ylabel('Error Absoluto (AE)')
            ax.set_title(lang, fontweight='bold', color=LANG_PAL[lang])

        sys_p = [mpatches.Patch(color=v, label=k) for k, v in SYS_PAL.items()]
        fig.legend(handles=sys_p, title='Sistema', loc='lower center',
                   ncol=2, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle('H2 — salina.web vs salina.ai4life por idioma (Mann-Whitney U)',
                     fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Barras de medianas
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(langs_activos)); w = 0.35
        meds_w = [dq_f[dq_f['idioma']==l]['ae_web'].median() for l in langs_activos]
        meds_a = [dq_f[dq_f['idioma']==l]['ae_ai4'].median() for l in langs_activos]
        bw = ax.bar(x-w/2, meds_w, w, label=WEB,  color=SYS_PAL[WEB], alpha=0.85)
        ba = ax.bar(x+w/2, meds_a, w, label=AI4,  color=SYS_PAL[AI4], alpha=0.85)
        for bar in list(bw)+list(ba):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(langs_activos)
        ax.set_ylabel('Mediana del Error Absoluto')
        ax.set_title('Mediana AE por idioma y sistema', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
#  SECCIÓN CUALITATIVA
# ════════════════════════════════════════════════════════════════════════════════
elif seccion == "📝 Cualitativo (LaBSE)":

    st.title("📝 Análisis Cualitativo — Fallos de recuperación")
    st.info(
        "ℹ️ El cálculo de embeddings LaBSE requiere ejecutar el notebook localmente "
        "(modelo ~500MB). Este dashboard muestra el análisis de **fallos de recuperación** "
        "(respuestas negativas) y estadísticos descriptivos de las respuestas textuales, "
        "que no requieren el modelo.",
        icon="ℹ️"
    )

    dc_f = dc[dc['idioma'].isin(langs_activos)]

    # ── Métricas globales ─────────────────────────────────────────────────────
    st.subheader("Fallos de recuperación detectados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros cualitativos", len(dc_f))
    c2.metric("Fallos salina.web",
              f"{dc_f['neg_web'].sum()} ({dc_f['neg_web'].mean()*100:.0f}%)")
    c3.metric("Fallos salina.ai4life",
              f"{dc_f['neg_ai4'].sum()} ({dc_f['neg_ai4'].mean()*100:.0f}%)")
    c4.metric("Longitud media resp. web",
              f"{dc_f['resp_web'].str.len().mean():.0f} chars")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📉 Fallos por idioma y categoría",
        "📋 Explorador de respuestas",
        "📐 Longitud de respuestas"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Tasa de fallos por idioma**")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, (sn, nc) in zip(axes, [(WEB,'neg_web'),(AI4,'neg_ai4')]):
                rates = [dc_f[dc_f['idioma']==l][nc].mean()*100
                         for l in langs_activos]
                bars  = ax.bar(langs_activos, rates,
                               color=[LANG_PAL[l] for l in langs_activos],
                               alpha=0.85, edgecolor='white')
                for bar, rate in zip(bars, rates):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.5, f'{rate:.0f}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.set_ylim(0, max(rates)*1.4+5)
                ax.set_ylabel('Tasa de fallo (%)')
                ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
            fig.suptitle('Fallos de recuperación por idioma', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Fallos por categoría**")
            rows = []
            for cat in CATS_CUAL:
                sub = dc_f[dc_f['categoria']==cat]
                rows.append({
                    'Categoría': cat,
                    'Fallos web': f"{sub['neg_web'].sum()} ({sub['neg_web'].mean()*100:.0f}%)",
                    'Fallos ai4life': f"{sub['neg_ai4'].sum()} ({sub['neg_ai4'].mean()*100:.0f}%)",
                    'n': len(sub)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("**Fallos por idioma y categoría (heatmap)**")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            for ax, (sn, nc) in zip(axes, [(WEB,'neg_web'),(AI4,'neg_ai4')]):
                pivot = dc_f.pivot_table(
                    values=nc, index='idioma', columns='categoria',
                    aggfunc='mean').round(2) * 100
                pivot = pivot.loc[
                    [l for l in LANGS if l in pivot.index],
                    [c for c in CATS_CUAL if c in pivot.columns]
                ]
                sns.heatmap(pivot, ax=ax, annot=True, fmt='.0f', cmap='Reds',
                            vmin=0, vmax=100,
                            linewidths=0.5, linecolor='#eee',
                            cbar_kws={'label': 'Tasa fallo (%)'})
                ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
                ax.tick_params(axis='x', rotation=20)
            fig.suptitle('Tasa de fallos (%) por idioma y categoría', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("**Explorador de respuestas**")
        cat_sel  = st.selectbox("Categoría", CATS_CUAL)
        lang_sel = st.selectbox("Idioma", langs_activos)
        tipo_sel = st.radio("Mostrar", ["Todas", "Solo fallos", "Solo exitosas"],
                            horizontal=True)

        sub = dc_f[(dc_f['categoria']==cat_sel) & (dc_f['idioma']==lang_sel)]
        if tipo_sel == "Solo fallos":
            sub = sub[sub['neg_web'] | sub['neg_ai4']]
        elif tipo_sel == "Solo exitosas":
            sub = sub[~sub['neg_web'] & ~sub['neg_ai4']]

        st.markdown(f"*{len(sub)} registros*")

        for _, row in sub.iterrows():
            with st.expander(f"🔍 {row['pregunta'][:80]}..."):
                c1, c2 = st.columns(2)
                with c1:
                    fallo = "🔴 FALLO" if row['neg_web'] else "🟢 OK"
                    st.markdown(f"**{WEB}** {fallo}")
                    st.markdown(str(row['resp_web'])[:600])
                with c2:
                    fallo = "🔴 FALLO" if row['neg_ai4'] else "🟢 OK"
                    st.markdown(f"**{AI4}** {fallo}")
                    st.markdown(str(row['resp_ai4'])[:600])

    with tab3:
        st.markdown("**Longitud de respuestas por idioma y sistema**")
        dc_f2 = dc_f.copy()
        dc_f2['len_web'] = dc_f2['resp_web'].str.len()
        dc_f2['len_ai4'] = dc_f2['resp_ai4'].str.len()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (sn, col) in zip(axes, [(WEB,'len_web'),(AI4,'len_ai4')]):
            data = [dc_f2[dc_f2['idioma']==l][col].values for l in langs_activos]
            bp   = ax.boxplot(data, patch_artist=True,
                              medianprops=dict(color='black', linewidth=2))
            for patch, lang in zip(bp['boxes'], langs_activos):
                patch.set_facecolor(LANG_PAL[lang]); patch.set_alpha(0.78)
            ax.set_xticks(range(1, len(langs_activos)+1))
            ax.set_xticklabels(langs_activos)
            ax.set_ylabel('Longitud (caracteres)')
            ax.set_title(sn, fontweight='bold', color=SYS_PAL[sn])
        fig.suptitle('Longitud de respuestas por idioma y sistema', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("*Respuestas más largas no implican mayor calidad, "
                    "pero respuestas muy cortas pueden indicar fallos de recuperación "
                    "no detectados por el regex.*")

# ════════════════════════════════════════════════════════════════════════════════
#  RESUMEN EJECUTIVO
# ════════════════════════════════════════════════════════════════════════════════
elif seccion == "📊 Resumen ejecutivo":

    st.title("📊 Resumen Ejecutivo — Milestone 3 Bloque 1")
    st.markdown(f"**Sistemas evaluados:** {WEB} vs {AI4} | **α = {ALPHA}**")

    dq_f = dq[dq['idioma'].isin(langs_activos)]
    dc_f = dc[dc['idioma'].isin(langs_activos)]

    st.subheader("A. Rendimiento cuantitativo (MAE)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE global web",
              f"{dq_f['ae_web'].mean():.2f}",
              delta=f"Med={dq_f['ae_web'].median():.1f}")
    c2.metric("MAE global ai4life",
              f"{dq_f['ae_ai4'].mean():.2f}",
              delta=f"Med={dq_f['ae_ai4'].median():.1f}")
    c3.metric("Aciertos exactos web",
              f"{(dq_f['ae_web']==0).mean()*100:.0f}%")
    c4.metric("Aciertos exactos ai4life",
              f"{(dq_f['ae_ai4']==0).mean()*100:.0f}%")

    st.subheader("B. Tests estadísticos — síntesis")

    rows = []
    # H1
    for sn, col in [(WEB,'ae_web'),(AI4,'ae_ai4')]:
        grupos = [dq_f[dq_f['idioma']==l][col].values for l in langs_activos]
        if len(grupos) >= 2:
            H, p   = kruskal(*grupos)
            eps2   = epsilon_squared(H, len(grupos), sum(len(g) for g in grupos))
            rows.append({
                'Hipótesis': 'H1', 'Test': 'Kruskal-Wallis',
                'Sistema / Comparación': sn,
                'Estadístico': f'H={H:.3f}', 'p': f'{p:.4f}',
                'Efecto': f'ε²={eps2:.4f} ({eff_eps(eps2)})',
                'Decisión': '✓ Rechazar H₀' if p < ALPHA else '✗ No rechazar H₀'
            })

    # H2
    for lang in langs_activos:
        sub = dq_f[dq_f['idioma']==lang]
        g_w = sub['ae_web'].values; g_a = sub['ae_ai4'].values
        U, p = mannwhitneyu(g_w, g_a, alternative='two-sided')
        r    = rank_biserial_r(U, len(g_w), len(g_a))
        rows.append({
            'Hipótesis': 'H2', 'Test': 'Mann-Whitney U',
            'Sistema / Comparación': lang,
            'Estadístico': f'U={U:.1f}', 'p': f'{p:.4f}',
            'Efecto': f'r={r:.3f} ({eff_r(r)})',
            'Decisión': '✓ Rechazar H₀' if p < ALPHA else '✗ No rechazar H₀'
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("C. Fallos de recuperación (cualitativo)")
    c1, c2 = st.columns(2)
    c1.metric("Fallos salina.web",
              f"{dc_f['neg_web'].sum()}/{len(dc_f)} ({dc_f['neg_web'].mean()*100:.0f}%)")
    c2.metric("Fallos salina.ai4life",
              f"{dc_f['neg_ai4'].sum()}/{len(dc_f)} ({dc_f['neg_ai4'].mean()*100:.0f}%)")

    st.subheader("D. Limitación metodológica")
    st.warning(
        "**Diseño de caja negra:** Los resultados reflejan el comportamiento observable "
        "del sistema RAG completo (retriever + LLM generador). No es posible atribuir "
        "causalmente las diferencias a un componente específico.",
        icon="⚠️"
    )

    st.subheader("E. Vista completa de datos")
    with st.expander("Cuantitativo (MAE)"):
        st.dataframe(
            dq_f[['idioma','categoria','web','ai4','manual','ae_web','ae_ai4']]
                .rename(columns={'web': WEB, 'ai4': AI4,
                                 'ae_web': f'AE {WEB}', 'ae_ai4': f'AE {AI4}'}),
            use_container_width=True
        )
    with st.expander("Cualitativo (respuestas)"):
        st.dataframe(
            dc_f[['idioma','categoria','pregunta','neg_web','neg_ai4']]
                .rename(columns={'neg_web': f'Fallo {WEB}',
                                 'neg_ai4': f'Fallo {AI4}'}),
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "SALiNA Evaluation Dashboard · Master in Data Science for Social Sciences · "
    "Universidade de Aveiro · Milestone 3 — Bloque 1"
)