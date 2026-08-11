import pandas as pd
import numpy as np


def segmentar_clientes_rfm(
    df: pd.DataFrame,
    id_col: str = "ID_Cliente",
    date_col: str = "Data_da_fatura",
    value_col: str = "Valor",
    invoice_col: str | None = "N°_da_fatura",
    snapshot_date=None,
    n_bins: int = 5,
    keep_customer_cols: bool = True,
):
    """
    Segmenta clientes usando RFM de forma robusta.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base transacional.
    id_col : str
        Coluna identificadora do cliente.
    date_col : str
        Coluna de data da transação.
    value_col : str
        Coluna de valor monetário da transação.
    invoice_col : str | None
        Coluna de fatura/pedido para contar frequência por compra única.
        Se não existir, a frequência será contada por linhas do dataframe.
    snapshot_date : datetime-like | None
        Data de referência para calcular recência.
        Se None, usa a maior data da base.
    n_bins : int
        Número de faixas para scoring. Normalmente 4 ou 5.
    keep_customer_cols : bool
        Se True, preserva as demais colunas do cliente pegando a primeira ocorrência.

    Retorno
    -------
    pd.DataFrame
        DataFrame com Recencia, Frequencia, Monetario, scores e Categoria.
    """

    # Validação básica
    required = [id_col, date_col, value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    base = df.copy()

    # Conversões e limpeza
    base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
    base = base.dropna(subset=[id_col, date_col, value_col])

    # Data de referência da recência
    if snapshot_date is None:
        snapshot_date = base[date_col].max()
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    if pd.isna(snapshot_date):
        raise ValueError("Não foi possível definir a data de referência da recência.")

    # ------------------------------------------------------------------
    # Cálculo R, F, M
    # ------------------------------------------------------------------
    recencia = (
        base.groupby(id_col, as_index=False)[date_col]
        .max()
        .rename(columns={date_col: "Data_ultima_compra"})
    )
    recencia["Recencia"] = (snapshot_date - recencia["Data_ultima_compra"]).dt.days

    if invoice_col is not None and invoice_col in base.columns:
        frequencia = (
            base.groupby(id_col, as_index=False)[invoice_col]
            .nunique()
            .rename(columns={invoice_col: "Frequencia"})
        )
    else:
        frequencia = (
            base.groupby(id_col, as_index=False)[date_col]
            .count()
            .rename(columns={date_col: "Frequencia"})
        )

    monetario = (
        base.groupby(id_col, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "Monetario"})
    )

    rfm = recencia[[id_col, "Data_ultima_compra", "Recencia"]].merge(
        frequencia, on=id_col, how="inner"
    ).merge(
        monetario, on=id_col, how="inner"
    )

    # ------------------------------------------------------------------
    # Função interna para score por quantis, robusta a empates
    # ------------------------------------------------------------------
    def _score_quantil(serie: pd.Series, reverse: bool = False, bins: int = 5) -> pd.Series:
        s = pd.to_numeric(serie, errors="coerce")

        if s.nunique(dropna=True) == 0:
            return pd.Series(1, index=serie.index, dtype="int64")

        # reverse=True -> valores menores recebem score maior
        if reverse:
            s = -s

        # rank(method='first') reduz problemas com empates no qcut
        ranked = s.rank(method="first")

        q = min(bins, ranked.nunique())
        if q <= 1:
            return pd.Series(1, index=serie.index, dtype="int64")

        scores = pd.qcut(ranked, q=q, labels=False, duplicates="drop") + 1
        return scores.astype("int64")

    # R: quanto menor a recência, melhor a nota
    rfm["R_Score"] = _score_quantil(rfm["Recencia"], reverse=True, bins=n_bins)
    # F e M: quanto maior, melhor
    rfm["F_Score"] = _score_quantil(rfm["Frequencia"], reverse=False, bins=n_bins)
    rfm["M_Score"] = _score_quantil(rfm["Monetario"], reverse=False, bins=n_bins)

    # String final RFM
    rfm["RFM"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    # ------------------------------------------------------------------
    # Segmentação por regras, sem dicionário frágil com chaves repetidas
    # ------------------------------------------------------------------
    r = rfm["R_Score"]
    f = rfm["F_Score"]
    m = rfm["M_Score"]
    condicoes = [
        (r == 5) & (f >= 4),    # 1
        (r >= 4) & (f >= 4),    # 2
        (r >= 4) & (f == 3),    # 3
        (r >= 3) & (f >= 3),    # 4
        (r == 5) & (f <= 2),    # 5
        (r == 4) & (f <= 2),    # 6
        (r <= 2) & (f >= 4),    # 7
        (r <= 2) & (f <= 2),    # 8
        (r == 1) & (f == 1),    # 9
        (r == 1) & (f >= 4),    # 10
        (r == 3) & (f == 2),    # 11

    ]
    categorias = [
        "Campeões",             #1
        "Clientes Fiéis",       #2
        "Fiéis em Potencial",   #3
        "Clientes Ativos",      #4
        "Novos Clientes",       #5
        "Promessas",            #6
        "Em Risco",             #7
        "Hibernando",           #8
        "Perdidos",             #9
        "Não Pode Perder",      #10
        "Precisando de Atenção" #11
    ]
    rfm["Categoria"] = np.select(condicoes, categorias, default="Normais")

    rfm["Prioridade"] = np.select(
        [
            m >= 5,
            m >= 4,
            m >= 3,
        ],
        [
            "Premium",
            "Alta",
            "Normal"
        ],
        default="Baixa"
    )

    # ------------------------------------------------------------------
    # Preservar colunas do cliente, se desejado
    # ------------------------------------------------------------------
    if keep_customer_cols:
        # pega a primeira linha de cada cliente para manter colunas cadastrais
        customer_base = (
            base.sort_values([id_col, date_col])
            .drop_duplicates(subset=[id_col], keep="first")
        )

        # remove colunas transacionais que não fazem sentido no nível cliente
        drop_cols = [date_col, value_col]
        if invoice_col is not None and invoice_col in customer_base.columns:
            drop_cols.append(invoice_col)

        customer_base = customer_base.drop(columns=[c for c in drop_cols if c in customer_base.columns])

        rfm = customer_base.merge(rfm, on=id_col, how="left")

    # Ordenação final mais útil para análise
    ordem = [id_col, "Categoria", "Recencia", "Frequencia", "Monetario", "R_Score", "F_Score", "M_Score", "RFM"]
    ordem = [c for c in ordem if c in rfm.columns]
    restantes = [c for c in rfm.columns if c not in ordem]
    rfm = rfm[ordem + restantes]

    return rfm


## GERADO PELO CLAUDE:
# r = rfm["R_Score"]
# f = rfm["F_Score"]
# m = rfm["M_Score"]
#
# condicoes = [
#     # ── Cluster r alto (r = 4 ou 5) ──────────────────────────────
#     # Mais específico → mais amplo dentro do cluster
#     (r >= 4) & (f >= 4) & (m >= 4),   # Campeões
#     (r >= 4) & (f >= 4),               # Clientes Fiéis       (m qualquer)
#     (r >= 4) & (f >= 3) & (m >= 3),   # Fiéis em Potencial
#     (r >= 4) & (f >= 3),               # Clientes Ativos      ← preenche f=3, m≤2
#     (r >= 4) & (m >= 3),               # Novos Clientes       (f=1,2 com bom gasto)
#     (r >= 4),                          # Promessas            ← catch-all r≥4
#
#     # ── Cluster r médio (r = 3) ── antes caía 100% em "Outros" ──
#     (r == 3) & (f >= 4),               # Em Risco
#     (r == 3) & (m >= 3),               # Precisando de Atenção
#     (r == 3),                          # Promessas            ← catch-all r=3
#
#     # ── Cluster r baixo (r = 1 ou 2) ─────────────────────────────
#     (r <= 2) & (f >= 4) & (m >= 4),   # Não Pode Perder      ← agora acessível
#     (r <= 2) & (f >= 4),               # Em Risco
#     (r <= 2) & (m >= 4),               # Precisando de Atenção
#     (r <= 2) & (f >= 3),               # Em Risco             ← preenche f=3, m≤3
#     (r <= 2) & (f <= 2),               # Hibernando
# ]
#
# categorias = [
#     "Campeões",
#     "Clientes Fiéis",
#     "Fiéis em Potencial",
#     "Clientes Ativos",
#     "Novos Clientes",
#     "Promessas",           # catch-all r≥4
#     "Em Risco",
#     "Precisando de Atenção",
#     "Promessas",           # catch-all r=3
#     "Não Pode Perder",
#     "Em Risco",
#     "Precisando de Atenção",
#     "Em Risco",            # r≤2, f moderada
#     "Hibernando",
# ]
#
# rfm["Categoria"] = np.select(condicoes, categorias, default="Perdidos")