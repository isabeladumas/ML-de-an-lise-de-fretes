import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="E-Commerce Insights & Machine Learning",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada (moderna, limpa, espaçosa e visual)
st.markdown("""
<style>
    /* Tipografia e títulos principais em cor clara */
    .header-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 25px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important; /* Cor clara: Branco puro */
        margin-bottom: 0.35rem;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #93C5FD !important; /* Cor clara: Azul claro luminoso */
        margin-bottom: 0;
        line-height: 1.5;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        color: #38BDF8 !important; /* Cor clara: Ciano / Azul celeste */
        margin-top: 1.4rem;
        margin-bottom: 0.8rem;
    }
    
    /* Garantir cor clara em todos os títulos markdown */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
    }
    
    /* Cartões de métricas customizados */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Hero Banner para a Acurácia de Machine Learning */
    .accuracy-hero {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 25px;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.28);
    }
    .accuracy-title {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        color: #BFDBFE;
    }
    .accuracy-number {
        font-size: 3.4rem;
        font-weight: 900;
        line-height: 1.1;
        margin: 8px 0;
        color: #FFFFFF;
    }
    .accuracy-desc {
        font-size: 1.05rem;
        color: #EFF6FF;
        line-height: 1.5;
    }

    /* Cards explicativos coloridos para os quadrantes */
    .quad-card {
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        margin-bottom: 12px;
    }
    .quad-green {
        background-color: #ECFDF5;
        border: 1.5px solid #A7F3D0;
        color: #065F46;
    }
    .quad-red {
        background-color: #FEF2F2;
        border: 1.5px solid #FECACA;
        color: #991B1B;
    }
    .quad-blue {
        background-color: #F0F9FF;
        border: 1.5px solid #BAE6FD;
        color: #075985;
    }
    .quad-purple {
        background-color: #FAF5FF;
        border: 1.5px solid #E9D5FF;
        color: #5B21B6;
    }

    /* Regras de Ouro */
    .rule-card {
        background-color: #F8FAFC;
        border-left: 6px solid #2563EB;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CARREGAMENTO E PROCESSAMENTO DOS DADOS (CACHEADO)
# ==============================================================================
def find_data_path(filename):
    """Localiza o arquivo CSV em versao 3 ou no diretório atual."""
    candidates = [
        os.path.join("versao 3", filename),
        filename,
        os.path.join(os.path.dirname(__file__), "versao 3", filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename

@st.cache_data
def carregar_dados():
    items_path = find_data_path("olist_order_items_dataset.csv")
    products_path = find_data_path("olist_products_dataset.csv")
    payments_path = find_data_path("olist_order_payments_dataset.csv")

    df_items = pd.read_csv(items_path)
    df_products = pd.read_csv(products_path)
    df_payments = pd.read_csv(payments_path)

    # Limpeza e tratamento básico
    df_products = df_products.dropna()
    mediana_peso = df_products['product_weight_g'].median()
    df_products.loc[df_products['product_weight_g'] == 0, 'product_weight_g'] = mediana_peso
    df_payments = df_payments[df_payments['payment_installments'] > 0]

    # Pontos de Corte (Medianas do Mercado)
    mediana_preco = round(float(df_items['price'].median()), 2)
    mediana_frete = round(float(df_items['freight_value'].median()), 2)

    # Classificação em 4 Quadrantes
    def classificar_oferta(row):
        if row['price'] <= mediana_preco and row['freight_value'] <= mediana_frete:
            return 'Econômico (Baixo Preço / Baixo Frete)'
        elif row['price'] <= mediana_preco and row['freight_value'] > mediana_frete:
            return 'Gargalo (Baixo Preço / Alto Frete)'
        elif row['price'] > mediana_preco and row['freight_value'] <= mediana_frete:
            return 'Premium (Alto Preço / Baixo Frete)'
        else:
            return 'Carga Pesada (Alto Preço / Alto Frete)'

    df_items['segmento_oferta'] = df_items.apply(classificar_oferta, axis=1)
    df_items['freight_ratio'] = df_items['freight_value'] / df_items['price']

    # Enriquecimento com pagamentos e produtos
    pagamentos_pedidos = df_payments.groupby('order_id').agg({
        'payment_installments': 'mean'
    }).reset_index()

    df_completo = df_items.merge(pagamentos_pedidos, on='order_id', how='left')
    df_completo = df_completo.merge(
        df_products[['product_id', 'product_category_name', 'product_weight_g']],
        on='product_id',
        how='left'
    )

    # Rótulo de Target para o Machine Learning
    df_completo['alerta_gargalo'] = (
        (df_completo['price'] <= mediana_preco) & (df_completo['freight_value'] > mediana_frete)
    ).astype(int)

    # Tabela Executiva Consolidada
    tabela_executiva = df_completo.groupby('segmento_oferta').agg(
        total_itens=('order_id', 'count'),
        pct_volume=('order_id', lambda x: len(x) / len(df_completo) * 100),
        gmv_total=('price', 'sum'),
        ticket_medio_preco=('price', 'mean'),
        frete_medio=('freight_value', 'mean'),
        razao_frete_mediana=('freight_ratio', lambda x: x.median() * 100),
        parcelas_media=('payment_installments', 'mean'),
        peso_medio_kg=('product_weight_g', lambda x: x.mean() / 1000)
    ).reset_index()

    gmv_global = tabela_executiva['gmv_total'].sum()
    tabela_executiva['pct_gmv'] = (tabela_executiva['gmv_total'] / gmv_global) * 100
    tabela_executiva = tabela_executiva.sort_values(by='gmv_total', ascending=False).reset_index(drop=True)

    return df_completo, tabela_executiva, mediana_preco, mediana_frete

@st.cache_resource
def treinar_modelo(df_completo):
    # Features e Target
    X = df_completo[['price', 'product_weight_g']].copy()
    X.columns = ['Preço do Produto (R$)', 'Peso do Produto (g)']
    y = df_completo['alerta_gargalo']

    # Divisão Treino (75%) e Teste (25%)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Modelo Árvore de Decisão
    modelo = DecisionTreeClassifier(max_depth=2, random_state=42)
    modelo.fit(X_treino, y_treino)

    # Avaliação
    y_pred = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred) * 100
    matriz_conf = confusion_matrix(y_teste, y_pred)
    relatorio = classification_report(y_teste, y_pred, output_dict=True)

    return modelo, acuracia, X_teste, y_teste, matriz_conf, relatorio, list(X.columns)

# Carregamento
try:
    df_completo, tabela_executiva, mediana_preco, mediana_frete = carregar_dados()
    modelo_ml, acuracia_ml, X_teste, y_teste, matriz_conf, relatorio_ml, features_ml = treinar_modelo(df_completo)
except Exception as e:
    st.error(f"Erro ao carregar dados ou treinar o modelo: {e}")
    st.stop()


# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.markdown("## Inteligência de Vendas")
    st.caption("Diagnóstico Comercial & Machine Learning no E-Commerce")
    st.markdown("---")

    pagina = st.radio(
        "Navegue pelas Páginas:",
        [
            "1. Visão Geral: O Que Foi Feito no Projeto",
            "2. Diagnóstico: Onde Ganhamos e Perdemos Vendas",
            "3. Machine Learning: Previsão de Risco com 83.7% de Acurácia"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### Indicadores-Chave da Base")
    st.markdown(f"- **Total de Pedidos/Itens:** {len(df_completo):,}")
    st.markdown(f"- **Faturamento Total (GMV):** R$ {tabela_executiva['gmv_total'].sum():,.2f}")
    st.markdown(f"- **Ponto de Corte de Preço:** R$ {mediana_preco:.2f}")
    st.markdown(f"- **Ponto de Corte de Frete:** R$ {mediana_frete:.2f}")
    st.markdown(f"- **Acurácia do Machine Learning:** **{acuracia_ml:.1f}%**")

    st.markdown("---")
    st.info("**Objetivo:** Identificar com precisão quando o frete fica caro demais para o produto e evitar o abandono no checkout.")


# ==============================================================================
# PÁGINA 1: VISÃO GERAL & O QUE FOI FEITO
# ==============================================================================
if pagina == "1. Visão Geral: O Que Foi Feito no Projeto":
    st.markdown("""
    <div class="header-card">
        <div class="main-title">Segmentação Estratégica: O Peso do Frete nas Vendas</div>
        <div class="sub-title">Como a combinação de Preço do Produto e Custo do Frete determina a decisão de compra do cliente</div>
    </div>
    """, unsafe_allow_html=True)

    # O que foi feito em 3 blocos visuais simples
    st.markdown('<div class="section-header">O Que Foi Feito Neste Projeto? (Entenda em 3 Etapas)</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="metric-box">
            <span class="metric-label">Etapa 1: O Problema Real</span>
            <div style="font-size: 1.15rem; font-weight:700; margin: 8px 0; color:#DC2626;">Frete Caro Espanta no Checkout</div>
            <div style="font-size:0.92rem; color:#475569; line-height: 1.5;">
                O consumidor decide a compra pela soma total. Quando o frete custa quase o preço do produto, o cliente se sente lesado e <b>abandona o carrinho</b> imediatamente.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="metric-box">
            <span class="metric-label">Etapa 2: A Metodologia</span>
            <div style="font-size: 1.15rem; font-weight:700; margin: 8px 0; color:#2563EB;">Medianas de Mercado (Sem Ruído)</div>
            <div style="font-size:0.92rem; color:#475569; line-height: 1.5;">
                Usamos medianas reais de 112 mil pedidos (<b>R$ 74,99</b> no preço e <b>R$ 16,26</b> no frete). A mediana é imune a valores extremos, garantindo <b>regras 100% justas</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="metric-box">
            <span class="metric-label">Etapa 3: A Ação Estratégica</span>
            <div style="font-size: 1.15rem; font-weight:700; margin: 8px 0; color:#059669;">Prevenção com Machine Learning</div>
            <div style="font-size:0.92rem; color:#475569; line-height: 1.5;">
                Classificamos o catálogo em 4 quadrantes e treinamos um algoritmo simples e explicável que atinge <b>83.7% de acurácia</b> para alertar ofertas problemáticas antes da publicação.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Linha de métricas de topo
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total de Itens Analisados", value=f"{len(df_completo):,}")
    with m2:
        st.metric(label="Faturamento Total (GMV)", value=f"R$ {tabela_executiva['gmv_total'].sum()/1e6:.2f} M")
    with m3:
        st.metric(label="Ponto de Corte: Preço", value=f"R$ {mediana_preco:.2f}")
    with m4:
        st.metric(label="Ponto de Corte: Frete", value=f"R$ {mediana_frete:.2f}")

    st.markdown("---")

    # Os 4 Quadrantes e o Gráfico da Matriz
    st.markdown('<div class="section-header">Como o Catálogo Foi Dividido: Os 4 Perfis de Oferta</div>', unsafe_allow_html=True)
    st.write("A relação entre o Preço do Item e o Custo do Frete divide os produtos em 4 realidades comerciais bem distintas:")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("""
        <div class="quad-card quad-green">
            <b>1. Econômico</b><br>
            <small><b>33,1% do Catálogo</b></small><br><br>
            • <b>Preço Baixo / Frete Baixo</b><br>
            • Porta de entrada da plataforma<br>
            • Alto volume e compras por impulso
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class="quad-card quad-red">
            <b>2. Gargalo (Crítico)</b><br>
            <small><b>17,0% do Catálogo</b></small><br><br>
            • <b>Preço Baixo / Frete Alto</b><br>
            • Frete desproporcional ao produto<br>
            • Causa principal de perda de vendas
        </div>
        """, unsafe_allow_html=True)
    with q3:
        st.markdown("""
        <div class="quad-card quad-blue">
            <b>3. Premium</b><br>
            <small><b>17,0% do Catálogo</b></small><br><br>
            • <b>Preço Alto / Frete Baixo</b><br>
            • Altíssima margem de lucro<br>
            • O frete quase não pesa na decisão
        </div>
        """, unsafe_allow_html=True)
    with q4:
        st.markdown("""
        <div class="quad-card quad-purple">
            <b>4. Carga Pesada</b><br>
            <small><b>32,9% do Catálogo</b></small><br><br>
            • <b>Preço Alto / Frete Alto</b><br>
            • Produtos volumosos e de valor elevado<br>
            • Concentra 62,9% de toda a receita
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico da Matriz 2x2
    col_matriz, col_tabela = st.columns([1.1, 1.3])

    with col_matriz:
        st.markdown("#### Matriz 2x2: Onde Cada Pedido se Encontra")
        df_items_plot = df_completo.copy()
        df_items_plot['Grupo_Preco'] = df_items_plot['price'].apply(
            lambda x: f'Preço Alto (> R${mediana_preco:.2f})' if x > mediana_preco else f'Preço Baixo (<= R${mediana_preco:.2f})'
        )
        df_items_plot['Grupo_Frete'] = df_items_plot['freight_value'].apply(
            lambda y: f'Frete Alto (> R${mediana_frete:.2f})' if y > mediana_frete else f'Frete Baixo (<= R${mediana_frete:.2f})'
        )
        matriz_ofertas = pd.crosstab(
            df_items_plot['Grupo_Frete'],
            df_items_plot['Grupo_Preco'],
            normalize='all'
        ) * 100
        matriz_ofertas = matriz_ofertas.reindex(
            index=[f'Frete Alto (> R${mediana_frete:.2f})', f'Frete Baixo (<= R${mediana_frete:.2f})'],
            columns=[f'Preço Baixo (<= R${mediana_preco:.2f})', f'Preço Alto (> R${mediana_preco:.2f})']
        )

        fig_mat, ax_mat = plt.subplots(figsize=(6, 4.5), dpi=120)
        cmap = sns.light_palette('#1E88E5', as_cmap=True)
        sns.heatmap(
            matriz_ofertas,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=1.5,
            linecolor='white',
            ax=ax_mat
        )
        ax_mat.text(0.5, 0.72, 'GARGALO\n(Alto Risco)', ha='center', va='center', fontsize=9.5, color='#B91C1C', fontweight='bold')
        ax_mat.text(1.5, 0.72, 'CARGA PESADA\n(Alto GMV)', ha='center', va='center', fontsize=9.5, color='#4338CA', fontweight='bold')
        ax_mat.text(0.5, 1.72, 'ECONÔMICO\n(Alto Giro)', ha='center', va='center', fontsize=9.5, color='#047857', fontweight='bold')
        ax_mat.text(1.5, 1.72, 'PREMIUM\n(Alta Margem)', ha='center', va='center', fontsize=9.5, color='#0369A1', fontweight='bold')
        ax_mat.set_xlabel('Faixa de Preço do Produto', fontsize=10, fontweight='bold')
        ax_mat.set_ylabel('Faixa de Custo do Frete', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_mat)

    with col_tabela:
        st.markdown("#### Comparativo Financeiro dos 4 Segmentos")
        tab_show = tabela_executiva[['segmento_oferta', 'total_itens', 'pct_volume', 'gmv_total', 'pct_gmv', 'ticket_medio_preco', 'frete_medio']].copy()
        tab_show.columns = ['Segmento', 'Itens', 'Volume (%)', 'GMV (R$)', 'Faturamento (%)', 'Preço Médio', 'Frete Médio']
        tab_show['Segmento'] = tab_show['Segmento'].apply(lambda x: x.split(' (')[0])
        tab_show['Itens'] = tab_show['Itens'].apply(lambda x: f"{x:,}")
        tab_show['Volume (%)'] = tab_show['Volume (%)'].apply(lambda x: f"{x:.1f}%")
        tab_show['GMV (R$)'] = tab_show['GMV (R$)'].apply(lambda x: f"R$ {x:,.2f}")
        tab_show['Faturamento (%)'] = tab_show['Faturamento (%)'].apply(lambda x: f"{x:.1f}%")
        tab_show['Preço Médio'] = tab_show['Preço Médio'].apply(lambda x: f"R$ {x:.2f}")
        tab_show['Frete Médio'] = tab_show['Frete Médio'].apply(lambda x: f"R$ {x:.2f}")
        st.dataframe(tab_show, use_container_width=True, hide_index=True)

        st.markdown("""
        > **Conclusão Rápida:**  
        > O segmento **Gargalo** representa 17% dos pedidos mas apenas **6,4% do faturamento total**, pois o frete desproporcional estrangula a taxa de conversão no checkout.
        """)


# ==============================================================================
# PÁGINA 2: INSIGHTS & DIAGNÓSTICO DE NEGÓCIO
# ==============================================================================
elif pagina == "2. Diagnóstico: Onde Ganhamos e Perdemos Vendas":
    st.markdown("""
    <div class="header-card">
        <div class="main-title">Diagnóstico Comercial: Onde Estão os Lucros e as Perdas?</div>
        <div class="sub-title">Entenda de onde vem a receita, por que o frete espanta clientes e quais produtos sofrem mais</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Volume vs Faturamento (GMV)
    st.markdown('<div class="section-header">1. Volume de Pedidos vs. Faturamento Real (GMV): De Onde Vem o Dinheiro?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])

    with c1:
        fig_vol, ax_vol = plt.subplots(figsize=(8, 4.2), dpi=120)
        rotulos_curtos = [s.split(' (')[0] for s in tabela_executiva['segmento_oferta']]
        x = np.arange(len(rotulos_curtos))
        largura = 0.35

        bar1 = ax_vol.bar(x - largura/2, tabela_executiva['pct_volume'], width=largura, label='% do Volume (Demanda)', color='#10B981')
        bar2 = ax_vol.bar(x + largura/2, tabela_executiva['pct_gmv'], width=largura, label='% do Faturamento (GMV)', color='#F59E0B')

        for i in range(len(tabela_executiva)):
            ax_vol.text(x[i] - largura/2, tabela_executiva['pct_volume'][i] + 1.2, f"{tabela_executiva['pct_volume'][i]:.1f}%", ha='center', fontweight='bold', fontsize=9)
            ax_vol.text(x[i] + largura/2, tabela_executiva['pct_gmv'][i] + 1.2, f"{tabela_executiva['pct_gmv'][i]:.1f}%", ha='center', fontweight='bold', fontsize=9)

        ax_vol.set_xticks(x)
        ax_vol.set_xticklabels(rotulos_curtos, fontsize=10, fontweight='bold')
        ax_vol.set_ylabel('Percentual do Total (%)', fontsize=9.5)
        ax_vol.set_ylim(0, max(tabela_executiva['pct_gmv']) + 12)
        ax_vol.legend(frameon=True, fontsize=9.5)
        ax_vol.set_title('Comparativo: Quantidade de Pedidos vs. Valor Faturado', fontsize=11.5, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig_vol)

    with c2:
        st.markdown("""
        <div class="metric-box" style="margin-top: 15px;">
            <div style="font-size: 1.05rem; font-weight:700; color:#1E293B; margin-bottom:8px;">Diagnóstico do Faturamento:</div>
            <ul style="font-size:0.92rem; color:#475569; padding-left: 18px; line-height: 1.6;">
                <li><b>Carga Pesada Sustenta a Empresa:</b> Responde por 32,9% dos itens, mas gera <b>62,9% de toda a receita (R$ 8,55 milhões)</b>.</li>
                <li><b>Gargalo Sofre Estrangulamento:</b> Tem 17% dos itens, mas gera míseros <b>6,4% do dinheiro</b>.</li>
                <li><b>Econômico e Premium:</b> Garantem alto volume de visitantes e excelente rentabilidade unitária.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. A Barreira do Frete
    st.markdown('<div class="section-header">2. A Barreira Psicológica do Frete: Quanto Ele Pesa no Bolso do Consumidor?</div>', unsafe_allow_html=True)
    col_frete_g, col_frete_t = st.columns([1.5, 1])

    with col_frete_g:
        fig_fr, ax_fr = plt.subplots(figsize=(8, 4.2), dpi=120)
        ordem_razao = tabela_executiva.sort_values(by='razao_frete_mediana', ascending=False)
        rotulos_razao = [s.split(' (')[0] for s in ordem_razao['segmento_oferta']]
        cores_razao = ['#EF4444', '#10B981', '#8B5CF6', '#0284C7']

        bars = ax_fr.bar(rotulos_razao, ordem_razao['razao_frete_mediana'], color=cores_razao, width=0.5)
        for bar, v in zip(bars, ordem_razao['razao_frete_mediana']):
            ax_fr.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=10.5)

        ax_fr.axhline(30, color='#64748B', linestyle='--', alpha=0.8, label='Limite Tolerável de Mercado (30%)')
        ax_fr.set_ylabel('Frete como % do Preço do Produto', fontsize=9.5)
        ax_fr.set_ylim(0, 55)
        ax_fr.set_title('Peso do Frete no Valor do Item (Razão Mediana Frete / Preço)', fontsize=11.5, fontweight='bold', pad=10)
        ax_fr.legend(frameon=True, loc='upper right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_fr)

    with col_frete_t:
        st.markdown("""
        <div class="metric-box" style="margin-top: 15px;">
            <div style="font-size: 1.05rem; font-weight:700; color:#1E293B; margin-bottom:8px;">Por Que o Frete Espanta o Cliente?</div>
            <ul style="font-size:0.92rem; color:#475569; padding-left: 18px; line-height: 1.6;">
                <li><b>No Gargalo:</b> O frete consome <b>46,4% do valor do produto</b> (quase metade da compra!). O consumidor sente que está pagando mais pelo frete do que pelo item.</li>
                <li><b>No Premium:</b> O frete representa só <b>10,5%</b>, tornando a compra quase irresistível.</li>
                <li><b>Alerta Vermelho:</b> Qualquer oferta acima da linha de <b>30%</b> tem alta taxa de desistência.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Categorias Críticas no Gargalo
    st.markdown('<div class="section-header">3. Categorias Mais Prejudicadas: Onde as Vendas Estão Travadas no "Gargalo"?</div>', unsafe_allow_html=True)
    col_cat_g, col_cat_t = st.columns([1.5, 1])

    with col_cat_g:
        gargalo_df = df_completo[df_completo['segmento_oferta'].str.contains('Gargalo', na=False)]
        top_cat = gargalo_df['product_category_name'].value_counts().head(8)
        nomes_amigaveis = {
            'cama_mesa_banho': 'Cama, Mesa e Banho',
            'moveis_decoracao': 'Móveis e Decoração',
            'utilidades_domesticas': 'Utilidades Domésticas',
            'beleza_saude': 'Beleza e Saúde',
            'ferramentas_jardim': 'Ferramentas e Jardim',
            'esporte_lazer': 'Esporte e Lazer',
            'informatica_acessorios': 'Informática e Acessórios',
            'telefonia': 'Telefonia'
        }
        rotulos_cat = [nomes_amigaveis.get(c, c.replace('_', ' ').title()) for c in top_cat.index]

        fig_cat, ax_cat = plt.subplots(figsize=(8, 4.2), dpi=120)
        cores_cat = sns.color_palette("Reds_r", len(top_cat))
        bars_cat = ax_cat.barh(rotulos_cat[::-1], top_cat.values[::-1], color=cores_cat[::-1], height=0.6)

        for bar, v in zip(bars_cat, top_cat.values[::-1]):
            ax_cat.text(v + 35, bar.get_y() + bar.get_height()/2, f"{v:,} itens", va='center', fontweight='bold', fontsize=9, color='#B91C1C')

        ax_cat.set_xlabel('Quantidade de Itens Presos no Gargalo', fontsize=9.5)
        ax_cat.set_xlim(0, max(top_cat.values) + 350)
        ax_cat.set_title('Top 8 Categorias Que Mais Sofrem com Frete Desproporcional', fontsize=11.5, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig_cat)

    with col_cat_t:
        st.markdown("""
        <div class="metric-box" style="margin-top: 15px;">
            <div style="font-size: 1.05rem; font-weight:700; color:#1E293B; margin-bottom:8px;">Onde Agir com Prioridade?</div>
            <div style="font-size:0.92rem; color:#475569; line-height: 1.6;">
                Itens de <b>Cama, Mesa e Banho</b> e <b>Utilidades Domésticas</b> custam barato na unidade (ex: lençol, panela, travessa), mas pesam para o frete.<br><br>
                <b>Solução Prática:</b> Incentivar o lojista a vender em <b>Kits e Combos</b> (ex: Kit com 3 toalhas). O ticket sobe e o frete percentual despenca!
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Plano de Ação Estratégico
    st.markdown('<div class="section-header">4. Plano de Ação Comercial: Como Destravar as Vendas em Cada Segmento?</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("""
        <div class="quad-card quad-red">
            <h4 style="margin-top:0;">1. Gargalo</h4>
            <b>Problema:</b> Frete consome 46,4% do valor.<br><br>
            <b>Ações Imediatas:</b>
            <ul style="padding-left:16px; margin-top:8px;">
                <li><b>Kits e Combos:</b> Pacote com 2 ou 3 unidades eleva o valor para R$ 90 e o frete cai para menos de 20%.</li>
                <li><b>Teto de Frete:</b> Subsidiar pontualmente fretes acima de 30%.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="quad-card quad-blue">
            <h4 style="margin-top:0;">2. Premium</h4>
            <b>Problema:</b> Acelerar volume em ofertas lucrativas.<br><br>
            <b>Ações Imediatas:</b>
            <ul style="padding-left:16px; margin-top:8px;">
                <li><b>Selo Frete Grátis:</b> O frete médio é de apenas R$ 12, sendo facilmente absorvido pela margem.</li>
                <li><b>Vitrine Principal:</b> Destaque no topo da busca para máxima conversão.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="quad-card quad-green">
            <h4 style="margin-top:0;">3. Econômico</h4>
            <b>Problema:</b> Aumentar o valor médio do carrinho.<br><br>
            <b>Ações Imediatas:</b>
            <ul style="padding-left:16px; margin-top:8px;">
                <li><b>Gatilho de Carrinho:</b> <i>"Adicione mais R$ 25 para ganhar frete grátis"</i>.</li>
                <li><b>Compra em 1 Clique:</b> Fluxo ultra-rápido para incentivar compra por impulso.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown("""
        <div class="quad-card quad-purple">
            <h4 style="margin-top:0;">4. Carga Pesada</h4>
            <b>Problema:</b> Viabilizar compras caras e volumosas.<br><br>
            <b>Ações Imediatas:</b>
            <ul style="padding-left:16px; margin-top:8px;">
                <li><b>Parcelamento sem Juros (10x-12x):</b> Dilui o frete alto na parcela mensal.</li>
                <li><b>Frete LTL Corporativo:</b> Contratos dedicados para transporte de cargas pesadas.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PÁGINA 3: MACHINE LEARNING & PREVENÇÃO (PÁGINA DEDICADA)
# ==============================================================================
elif pagina == "3. Machine Learning: Previsão de Risco com 83.7% de Acurácia":
    st.markdown("""
    <div class="header-card">
        <div class="main-title">Machine Learning: Previsão Inteligente de Abandono no Checkout</div>
        <div class="sub-title">Algoritmo preditivo que avisa o lojista se o produto tem risco de encalhar antes mesmo de ser publicado</div>
    </div>
    """, unsafe_allow_html=True)

    # HERO BANNER: ACURÁCIA DO MODELO EM DESTAQUE CLARO
    st.markdown(f"""
    <div class="accuracy-hero">
        <div class="accuracy-title">DESEMPENHO COMPROVADO DO MODELO PREDITIVO</div>
        <div class="accuracy-number">{acuracia_ml:.1f}% de Acurácia Geral</div>
        <div class="accuracy-desc">
            Validado cientificamente em <b>{len(X_teste):,} produtos reais</b> de teste cego que o algoritmo <b>nunca viu durante o treinamento</b> (Split 75% treino / 25% teste).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Cards Explicativos: Simples e sem complicação
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown("""
        <div class="metric-box">
            <span class="metric-label">1. Algoritmo Escolhido</span>
            <div style="font-size: 1.15rem; font-weight:700; color:#1E293B; margin: 6px 0;">Árvore de Decisão Simples</div>
            <div style="font-size:0.9rem; color:#475569; line-height: 1.5;">
                <b>100% Explicável (Sem Caixa-Preta):</b> Funciona como um fluxograma lógico de regras (SE... ENTÃO), permitindo explicar o motivo de qualquer alerta para o lojista.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        st.markdown("""
        <div class="metric-box">
            <span class="metric-label">2. Dados Necessários</span>
            <div style="font-size: 1.15rem; font-weight:700; color:#2563EB; margin: 6px 0;">Apenas Preço e Peso</div>
            <div style="font-size:0.9rem; color:#475569; line-height: 1.5;">
                Não exige dados complexos nem histórico prévio. O modelo só precisa do <b>Preço de Venda (R$)</b> e do <b>Peso do Item (g)</b>, já informados no cadastro.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_e3:
        st.markdown("""
        <div class="metric-box">
            <span class="metric-label">3. Aplicação Prática</span>
            <div style="font-size: 1.15rem; font-weight:700; color:#059669; margin: 6px 0;">Alerta Automático no Cadastro</div>
            <div style="font-size:0.9rem; color:#475569; line-height: 1.5;">
                Antes de colocar o anúncio no ar, o sistema orienta o vendedor a montar um <b>Combo</b> caso detecte risco de frete desproporcional.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # A "Regra de Ouro" Descoberta
    st.markdown('<div class="section-header">A "Regra de Ouro" Descoberta Pela Inteligência Artificial</div>', unsafe_allow_html=True)
    st.write("Ao analisar o histórico de 112 mil pedidos, o Machine Learning descobriu de forma autônoma a raiz física do problema:")

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("""
        <div class="rule-card">
            <div style="font-size: 1.1rem; font-weight: 700; color: #1E40AF; margin-bottom: 6px;">
                Regra 1: Preço > R$ 74,99 $\\rightarrow$ Oferta Saudável
            </div>
            <span style="font-size:0.92rem; color:#334155; line-height: 1.5;">
                Quando o produto custa mais de R$ 74,99, o ticket é alto o suficiente para que o frete nunca se torne uma barreira desproporcional. O item é classificado como seguro.
            </span>
        </div>
        """, unsafe_allow_html=True)
    with r_col2:
        st.markdown("""
        <div class="rule-card" style="border-left-color: #EF4444;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #DC2626; margin-bottom: 6px;">
                Regra 2: Preço $\\le$ R$ 74,99 e Peso > 972 g (~1 kg) $\\rightarrow$ Risco de Gargalo!
            </div>
            <span style="font-size:0.92rem; color:#334155; line-height: 1.5;">
                <b>O Ponto de Ruptura Logística:</b> Quando o produto é barato e pesa mais de 1 kg, ele salta de faixa de frete nas transportadoras. O frete fica quase o valor do produto e trava a compra!
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico da Árvore de Decisão
    st.markdown('<div class="section-header">Fluxograma Visual: O Passo a Passo da Decisão do Modelo</div>', unsafe_allow_html=True)
    fig_tree, ax_tree = plt.subplots(figsize=(10, 4.5), dpi=130)
    plot_tree(
        modelo_ml,
        feature_names=features_ml,
        class_names=['Oferta Saudável', 'Risco de Gargalo'],
        filled=True,
        rounded=True,
        fontsize=10,  # Inteiro corrigido para compatibilidade total
        ax=ax_tree
    )
    plt.title('Árvore de Decisão: Classificação Automática de Ofertas (Profundidade = 2)', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    st.pyplot(fig_tree)

    st.markdown("---")

    # SIMULADOR INTERATIVO
    st.markdown('<div class="section-header">Simulador em Tempo Real: Teste o Risco de um Produto Novo</div>', unsafe_allow_html=True)
    st.write("Altere os controles abaixo e veja como a Inteligência Artificial avalia a competitividade do anúncio na hora:")

    sim_col1, sim_col2 = st.columns([1, 1.3])

    with sim_col1:
        st.markdown("##### Dados do Produto a Cadastrar")
        input_preco = st.slider(
            "Preço de Venda do Produto (R$):",
            min_value=5.0,
            max_value=300.0,
            value=45.0,
            step=1.0,
            help="Valor cobrado pelo produto na vitrine"
        )
        input_peso = st.slider(
            "Peso Físico do Produto Embalado (gramas):",
            min_value=50,
            max_value=5000,
            value=1500,
            step=50,
            help="Peso físico embalado para envio aos Correios ou transportadora"
        )

        st.caption(f"Valor selecionado: **R$ {input_preco:.2f}** | Peso selecionado: **{input_peso/1000:.2f} kg** ({input_peso} g)")

    with sim_col2:
        st.markdown("##### Diagnóstico Automático do Machine Learning")
        # Previsão
        input_df = pd.DataFrame([[input_preco, input_peso]], columns=features_ml)
        predicao = modelo_ml.predict(input_df)[0]
        probabilidades = modelo_ml.predict_proba(input_df)[0]

        if predicao == 1:
            st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 2px solid #EF4444; border-radius: 12px; padding: 20px;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #DC2626;">
                    ALERTA: Risco Alto de Abandono de Carrinho!
                </div>
                <div style="font-size: 0.96rem; color: #7F1D1D; margin: 10px 0; line-height: 1.5;">
                    Este produto tem preço baixo (<b>R$ {input_preco:.2f}</b>) e peso superior a 1 kg (<b>{input_peso/1000:.2f} kg</b>).
                    O frete previsto consumirá mais de 40% do valor da compra, gerando atrito no checkout.
                </div>
                <hr style="border: 0; border-top: 1px solid #FECACA; margin: 12px 0;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #991B1B;">
                    O que o Lojista Deve Fazer:
                </div>
                <div style="font-size: 0.9rem; color: #475569; margin-top: 6px; line-height: 1.5;">
                    • <b>Montar um Kit:</b> Oferecer um combo com 2 ou 3 unidades para elevar o valor do carrinho acima de R$ 75,00.<br>
                    • <b>Venda Casada (Cross-Sell):</b> Incluir este item como acessório complementar em produtos de maior valor.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 12px; padding: 20px;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #059669;">
                    OFERTA SAUDÁVEL: Baixo Risco de Abandono
                </div>
                <div style="font-size: 0.96rem; color: #065F46; margin: 10px 0; line-height: 1.5;">
                    Este produto apresenta excelente proporção entre valor comercial (<b>R$ {input_preco:.2f}</b>) e peso logístico (<b>{input_peso/1000:.2f} kg</b>).
                </div>
                <hr style="border: 0; border-top: 1px solid #A7F3D0; margin: 12px 0;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #047857;">
                    O que o Lojista Deve Fazer:
                </div>
                <div style="font-size: 0.9rem; color: #475569; margin-top: 6px; line-height: 1.5;">
                    • <b>Aprovação Imediata:</b> O produto tem excelente atratividade no checkout!<br>
                    • <b>Oportunidade:</b> Se o preço for > R$ 75 e o peso leve, aplique o selo <b>Frete Grátis</b> para acelerar as vendas.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Validação e Métricas Científicas
    st.markdown('<div class="section-header">Indicadores Técnicos de Validação do Algoritmo</div>', unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric(label="Acurácia no Teste Cego", value=f"{acuracia_ml:.1f}%")
    with col_t2:
        st.metric(label="Quantidade de Produtos Validados", value=f"{len(X_teste):,}")
    with col_t3:
        st.metric(label="Níveis da Árvore de Decisão", value="2 (Simples & Explicável)")
