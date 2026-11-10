# Segmentação Estratégica de Ofertas no E-Commerce e Machine Learning

Projeto de inteligência de negócios e modelagem preditiva aplicado a dados públicos de e-commerce brasileiro (Olist). O objetivo central é diagnosticar e prevenir o abandono de carrinho decorrente de fretes desproporcionais ao valor dos produtos.

---

## 1. O Problema de Negócio

No checkout, o cliente não decide a compra olhando apenas o preço do produto: a conversão é definida pelo **custo total (Preço + Frete)**:
* **Baixo Preço + Alto Frete ("Gargalo")**: O frete custa quase a metade do valor do produto (mediana de **46,4%** do preço). Isso gera uma forte barreira psicológica e abandono imediato de carrinho.
* **Alto Preço + Alto Frete ("Carga Pesada")**: Representa 32,9% dos itens, mas concentra **62,9% de todo o faturamento (GMV)** (R$ 8,55 milhões).
* **Alto Preço + Baixo Frete ("Premium")**: O frete consome apenas **10,5%** do produto, com altíssima taxa de conversão.
* **Baixo Preço + Baixo Frete ("Econômico")**: Porta de entrada e compras por impulso (33,1% dos pedidos).

A segmentação foi estruturada a partir das **Medianas Empíricas de Mercado**:
* **Linha de Corte de Preço**: R$ 74,99
* **Linha de Corte de Frete**: R$ 16,26

---

## 2. Machine Learning Aplicado: Prevenção do Gargalo

Para evitar que anúncios com risco de abandono entrem no ar sem uma estratégia comercial adequada, foi desenvolvida uma **Árvore de Decisão** supervisionada:

* **Algoritmo**: `DecisionTreeClassifier(max_depth=2)` (100% explicável, sem caixa-preta)
* **Features de Entrada**: Preço do Produto (R$) e Peso do Produto (g)
* **Desempenho no Teste Cego**: **83.7% de Acurácia** validada em **28.163 produtos reais** (split 75/25)

### A Regra de Ouro Descoberta:
1. **Regra de Preço**: Produtos com valor **> R$ 74,99** não sofrem desse gargalo.
2. **Ponto de Ruptura Logística**: Para produtos com valor **<= R$ 74,99**, produtos que pesam mais de **972 gramas (~1 kg)** saltam de faixa tarifária de frete e tornam-se **Gargalos de Conversão**.

---

## 3. Aplicação Web Interativa (Streamlit)

O projeto inclui um dashboard interativo desenvolvido em Streamlit com 3 visões:
1. **Visão Geral**: Apresentação executiva, matriz 2x2 de segmentação e KPIs principais.
2. **Diagnóstico & Insights**: Análise de Volume vs. Faturamento, impacto psicológico do frete, categorias críticas e plano de ação comercial.
3. **Machine Learning & Prevenção**: Apresentação da acurácia do modelo (83.7%), fluxograma visual da árvore e **Simulador em Tempo Real** onde o vendedor testa qualquer produto informando preço e peso.

---

## 4. Como Executar Localmente

### Pré-requisitos
* Python 3.10 ou superior

### Instalação das Dependências
```bash
pip install -r requirements.txt
```

### Execução do Streamlit
```bash
python -m streamlit run app.py
```
Acesse no seu navegador: `http://localhost:8501`

---

## 5. Estrutura de Arquivos

```
├── app.py                                   # Aplicação Streamlit
├── requirements.txt                         # Dependências do projeto
├── README.md                                # Documentação do projeto
├── .gitignore                               # Arquivos ignorados pelo Git
└── versao 3/
    ├── projeto.ipynb                        # Notebook Jupyter com análises e ML
    ├── olist_order_items_dataset.csv        # Dados de itens e frete
    ├── olist_products_dataset.csv           # Dados físicos de produtos
    ├── olist_order_payments_dataset.csv     # Dados de pagamentos
    └── ...
```
