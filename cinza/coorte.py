import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_period_dtype



# 1. Criar a coluna de Coorte (primeira compra do cliente)
def criar_coorte(df, id_cliente: str, ano_mes: str):
    """
    Cria a coluna Coorte contendo o primeiro mês de compra
    de cada cliente.
    """
    df = df.copy()

    df["Coorte"] = (
        df.groupby(id_cliente)[ano_mes]
          .transform("min")
    )

    return df


# 2. Calcular o período da compra dentro da coorte
def calcular_periodo(df, ano_mes: str):
    """
    Calcula o período (1,2,3...) usando diferença real entre datas.
    """
    df = df.copy()

    df["Periodo"] = (
        (df[ano_mes].dt.year - df["Coorte"].dt.year) * 12
        + (df[ano_mes].dt.month - df["Coorte"].dt.month)
    )

    return df


# 3. Agrupar as informações da coorte
def agrupar_coortes(df, id_base: str, id_cliente: str, receita: str):
    agrupado = (
        df
        .groupby(["Coorte", "Periodo"])
        .agg(
            Total_clientes=(id_cliente, "nunique"),
            Total_faturas=(id_base, "nunique"),
            Receita=(receita, "sum")
        )
    )
    return agrupado


# 4. Calcular matriz de retenção
def calcular_retencao(cohorts):
    clientes_base = (
        cohorts["Total_clientes"]
        .groupby(level=0)
        .first()
    )

    matriz_clientes = cohorts["Total_clientes"].unstack("Periodo")

    retencao = matriz_clientes.divide(clientes_base, axis=0)

    return retencao, matriz_clientes


# 5. Pipeline completo
def gerar_matriz_retencao(df, id_base: str, id_cliente: str, receita: str, ano_mes: str):
    df = df.copy()

    # Verifica se a coluna já é Period[M]
    if not is_period_dtype(df[ano_mes]):
        try:
            df[ano_mes] = (
                pd.to_datetime(df[ano_mes], format="%Y-%m")
                .dt.to_period("M")
            )
        except Exception as e:
            raise ValueError(f"Não possivel converter a coluna {ano_mes} em 'Period[M]' - [ERRO]: {e}")

    df = criar_coorte(df, id_cliente, ano_mes)
    df = calcular_periodo(df, ano_mes)
    cohorts = agrupar_coortes(df, id_base, id_cliente, receita)
    retencao, matriz_clientes = calcular_retencao(cohorts)

    return retencao, matriz_clientes, cohorts


# Heatmap de retenção
def coorte_heatmap(retencao, matriz_clientes):
    retencao_plot = retencao.copy()
    retencao_plot.columns = [f"M{c}" for c in retencao_plot.columns]
    plt.figure(figsize=(16, 8))

    # Tamanho de cada coorte usando o primeiro período disponível
    tamanho_coorte = matriz_clientes.iloc[:, 0]

    # Monta os rótulos da forma: 2021-01 | n=842
    y_labels = [
        f"{str(coorte)} | n={int(tamanho_coorte.loc[coorte])}"
        for coorte in retencao_plot.index
    ]

    sns.heatmap(
        retencao_plot,
        annot=True,
        fmt=".0%",
        cmap=sns.light_palette("#1f77b4", as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        mask=retencao_plot.isnull(),
        cbar_kws={"label": "Taxa de retenção"},
        yticklabels=y_labels
    )

    plt.title(
        "Análise de Coorte - Retenção de Clientes",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Período")
    plt.ylabel("Coorte")

    plt.tight_layout()
    plt.show()


# Heatmap de clientes
def clientes_heatmap(matriz_clientes):
    # renomeando os períodos para 'Mi'
    matriz_plot = matriz_clientes.copy()
    matriz_plot.columns = [f"M{c}" for c in matriz_plot.columns]

    plt.figure(figsize=(16, 8))

    # Tamanho da coorte (Período 1)
    tamanho_coorte = matriz_clientes.iloc[:, 0]

    # Rótulos do eixo Y
    y_labels = [
        f"{str(coorte)} | n={int(tamanho_coorte.loc[coorte])}"
        for coorte in matriz_clientes.index
    ]

    sns.heatmap(
        matriz_clientes,
        annot=True,
        fmt=".0f",
        cmap=sns.light_palette("#1f77b4", as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        mask=matriz_clientes.isnull(),
        cbar_kws={"label": "Quantidade de clientes"},
        yticklabels=y_labels
    )

    plt.title(
        "Análise de Coorte - Quantidade de Clientes",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Período")
    plt.ylabel("Coorte")

    plt.tight_layout()
    plt.show()