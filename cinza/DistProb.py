import numpy as np
from scipy.stats import (
    norm,
    uniform,
    expon,
    t,
    chi2,
    f,
    beta,
    gamma,
    weibull_min,
    lognorm,
    binom,
    poisson,
    bernoulli,
    geom,
    hypergeom,
    nbinom,
    multinomial,
    multivariate_normal,
    dirichlet
)
from scipy.special import gamma as gamma_func





# ==========================================================
#                  FUNÇÕES AUXILIARES
# ==========================================================


# calcula a probabilidade/densidade para distribuições contínuas
def _prob_continua(dist, x, tipo, a, b):
    if tipo == "densidade":
        return dist.pdf(x)

    elif tipo in ("<=", "<"):
        return dist.cdf(x)

    elif tipo in (">=", ">"):
        return 1 - dist.cdf(x)

    elif tipo == "entre":
        return dist.cdf(b) - dist.cdf(a)

    else:
        raise ValueError(f"Tipo '{tipo}' inválido.")


# calcula a probabilidade para distribuições discretas
def _prob_discreta(dist, x, tipo, a, b):
    if tipo == "pontual":
        return dist.pmf(x)

    elif tipo == "<=":
        return dist.cdf(x)

    elif tipo == "<":
        return dist.cdf(x - 1)

    elif tipo == ">=":
        return 1 - dist.cdf(x - 1)

    elif tipo == ">":
        return 1 - dist.cdf(x)

    elif tipo == "entre":
        return dist.cdf(b) - dist.cdf(a - 1)

    else:
        raise ValueError(f"Tipo '{tipo}' inválido.")


# monta o dicionário de resultado padrão, com interpretação em texto
def _montar_resultado(
        nome_dist, parametros, x, tipo, valor,
        media_teor=None, var_teor=None, a=None, b=None
):
    dp_teor = (
        float(np.sqrt(var_teor))
        if var_teor is not None
        else None
    )

    if tipo == "densidade":
        interpretacao = (
            f"A densidade de probabilidade da distribuição {nome_dist} "
            f"em x = {x} é igual a {valor:.4f}."
        )

    elif tipo == "pontual":
        interpretacao = (
            f"A probabilidade pontual P(X = {x}) da distribuição {nome_dist} "
            f"é igual a {valor:.4f} ({valor * 100:.2f}%)."
        )

    elif tipo == "entre":
        interpretacao = (
            f"A probabilidade de X estar entre {a} e {b}, "
            f"segundo a distribuição {nome_dist}, "
            f"é igual a {valor:.4f} ({valor * 100:.2f}%)."
        )

    else:
        interpretacao = (
            f"A probabilidade P(X {tipo} {x}) da distribuição {nome_dist} "
            f"é igual a {valor:.4f} ({valor * 100:.2f}%)."
        )

    return {
        "distribuicao": nome_dist,
        "parametros": parametros,
        "x": x,
        "tipo": tipo,
        "probabilidade": float(valor),
        "media_teorica": media_teor,
        "variancia_teorica": var_teor,
        "dp_teorico": dp_teor,
        "interpretacao": interpretacao
    }




#############################################
#          DISTRIBUIÇÕES CONTÍNUAS          #
#############################################


# distribuição normal
def dist_normal(x=None, media=0, dp=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Normal (Gaussiana).

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    media : float
        Média (mu) da distribuição.
    dp : float
        Desvio padrão (sigma) da distribuição.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = norm(loc=media, scale=dp)

    valor = _prob_continua(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Normal",
        parametros={"media": media, "dp": dp},
        x=x, tipo=tipo, valor=valor,
        media_teor=media, var_teor=dp ** 2,
        a=a, b=b
    )


# distribuição uniforme contínua
def dist_uniforme(x=None, minimo=0, maximo=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Uniforme Contínua.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    minimo : float
        Limite inferior do intervalo.
    maximo : float
        Limite superior do intervalo.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = uniform(loc=minimo, scale=maximo - minimo)

    valor = _prob_continua(dist, x, tipo, a, b)

    media_teor = (minimo + maximo) / 2
    var_teor = (maximo - minimo) ** 2 / 12

    return _montar_resultado(
        nome_dist="Uniforme",
        parametros={"minimo": minimo, "maximo": maximo},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )


# distribuição exponencial
def dist_exponencial(x=None, taxa=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Exponencial.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    taxa : float
        Taxa de ocorrência (lambda).
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = expon(scale=1 / taxa)

    valor = _prob_continua(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Exponencial",
        parametros={"taxa": taxa},
        x=x, tipo=tipo, valor=valor,
        media_teor=1 / taxa, var_teor=1 / taxa ** 2,
        a=a, b=b
    )


# distribuição t de student
def dist_t_student(x=None, gl=1, tipo="densidade", a=None, b=None):
    """
    Distribuição t de Student.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    gl : int
        Graus de liberdade.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = t(df=gl)

    valor = _prob_continua(dist, x, tipo, a, b)

    media_teor = 0 if gl > 1 else None
    var_teor = gl / (gl - 2) if gl > 2 else None

    return _montar_resultado(
        nome_dist="t-Student",
        parametros={"gl": gl},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )


# distribuição qui-quadrado
def dist_qui_quadrado(x=None, gl=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Qui-Quadrado.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    gl : int
        Graus de liberdade.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = chi2(df=gl)

    valor = _prob_continua(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Qui-Quadrado",
        parametros={"gl": gl},
        x=x, tipo=tipo, valor=valor,
        media_teor=gl, var_teor=2 * gl,
        a=a, b=b
    )


# distribuição f de snedecor
def dist_f(x=None, gl1=1, gl2=1, tipo="densidade", a=None, b=None):
    """
    Distribuição F de Snedecor.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    gl1 : int
        Graus de liberdade do numerador.
    gl2 : int
        Graus de liberdade do denominador.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = f(dfn=gl1, dfd=gl2)

    valor = _prob_continua(dist, x, tipo, a, b)

    media_teor = gl2 / (gl2 - 2) if gl2 > 2 else None

    var_teor = (
        (2 * gl2 ** 2 * (gl1 + gl2 - 2))
        / (gl1 * (gl2 - 2) ** 2 * (gl2 - 4))
    ) if gl2 > 4 else None

    return _montar_resultado(
        nome_dist="F",
        parametros={"gl1": gl1, "gl2": gl2},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )


# distribuição beta
def dist_beta(x=None, alfa=1, beta_par=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Beta.

    Parameters
    ----------
    x : float
        Valor de referência (entre 0 e 1) para o cálculo.
    alfa : float
        Parâmetro de forma alfa.
    beta_par : float
        Parâmetro de forma beta.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = beta(a=alfa, b=beta_par)

    valor = _prob_continua(dist, x, tipo, a, b)

    media_teor = alfa / (alfa + beta_par)

    var_teor = (
        (alfa * beta_par)
        / ((alfa + beta_par) ** 2 * (alfa + beta_par + 1))
    )

    return _montar_resultado(
        nome_dist="Beta",
        parametros={"alfa": alfa, "beta": beta_par},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )


# distribuição gamma
def dist_gamma(x=None, forma=1, escala=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Gamma.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    forma : float
        Parâmetro de forma (k).
    escala : float
        Parâmetro de escala (theta).
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = gamma(a=forma, scale=escala)

    valor = _prob_continua(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Gamma",
        parametros={"forma": forma, "escala": escala},
        x=x, tipo=tipo, valor=valor,
        media_teor=forma * escala, var_teor=forma * escala ** 2,
        a=a, b=b
    )


# distribuição weibull
def dist_weibull(x=None, forma=1, escala=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Weibull.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo.
    forma : float
        Parâmetro de forma (k).
    escala : float
        Parâmetro de escala (lambda).
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = weibull_min(c=forma, scale=escala)

    valor = _prob_continua(dist, x, tipo, a, b)

    media_teor = escala * gamma_func(1 + 1 / forma)

    var_teor = escala ** 2 * (
        gamma_func(1 + 2 / forma) - gamma_func(1 + 1 / forma) ** 2
    )

    return _montar_resultado(
        nome_dist="Weibull",
        parametros={"forma": forma, "escala": escala},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )


# distribuição lognormal
def dist_lognormal(x=None, media_log=0, dp_log=1, tipo="densidade", a=None, b=None):
    """
    Distribuição Lognormal.

    Parameters
    ----------
    x : float
        Valor de referência para o cálculo (x > 0).
    media_log : float
        Média do logaritmo da variável.
    dp_log : float
        Desvio padrão do logaritmo da variável.
    tipo : str
        "densidade", "<=", "<", ">=", ">" ou "entre".
    a, b : float
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = lognorm(s=dp_log, scale=np.exp(media_log))

    valor = _prob_continua(dist, x, tipo, a, b)

    media_teor = np.exp(media_log + dp_log ** 2 / 2)

    var_teor = (
        (np.exp(dp_log ** 2) - 1)
        * np.exp(2 * media_log + dp_log ** 2)
    )

    return _montar_resultado(
        nome_dist="Lognormal",
        parametros={"media_log": media_log, "dp_log": dp_log},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )




#############################################
#          DISTRIBUIÇÕES DISCRETAS          #
#############################################


# distribuição binomial
def dist_binomial(x=None, n=1, p=0.5, tipo="pontual", a=None, b=None):
    """
    Distribuição Binomial.

    Parameters
    ----------
    x : int
        Número de sucessos.
    n : int
        Número de tentativas.
    p : float
        Probabilidade de sucesso em cada tentativa.
    tipo : str
        "pontual", "<=", "<", ">=", ">" ou "entre".
    a, b : int
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = binom(n=n, p=p)

    valor = _prob_discreta(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Binomial",
        parametros={"n": n, "p": p},
        x=x, tipo=tipo, valor=valor,
        media_teor=n * p, var_teor=n * p * (1 - p),
        a=a, b=b
    )


# distribuição de poisson
def dist_poisson(x=None, lam=1, tipo="pontual", a=None, b=None):
    """
    Distribuição de Poisson.

    Parameters
    ----------
    x : int
        Número de ocorrências.
    lam : float
        Taxa média de ocorrências (lambda).
    tipo : str
        "pontual", "<=", "<", ">=", ">" ou "entre".
    a, b : int
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = poisson(mu=lam)

    valor = _prob_discreta(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Poisson",
        parametros={"lambda": lam},
        x=x, tipo=tipo, valor=valor,
        media_teor=lam, var_teor=lam,
        a=a, b=b
    )


# distribuição de bernoulli
def dist_bernoulli(x=None, p=0.5, tipo="pontual"):
    """
    Distribuição de Bernoulli.

    Parameters
    ----------
    x : int
        Resultado (0 ou 1).
    p : float
        Probabilidade de sucesso (x = 1).
    tipo : str
        "pontual", "<=", "<", ">=" ou ">".

    Returns
    -------
    dict
    """

    dist = bernoulli(p=p)

    valor = _prob_discreta(dist, x, tipo, None, None)

    return _montar_resultado(
        nome_dist="Bernoulli",
        parametros={"p": p},
        x=x, tipo=tipo, valor=valor,
        media_teor=p, var_teor=p * (1 - p)
    )


# distribuição geométrica
def dist_geometrica(x=None, p=0.5, tipo="pontual", a=None, b=None):
    """
    Distribuição Geométrica.

    Modela o número de tentativas até o primeiro sucesso
    (suporte iniciando em 1).

    Parameters
    ----------
    x : int
        Número de tentativas até o primeiro sucesso.
    p : float
        Probabilidade de sucesso em cada tentativa.
    tipo : str
        "pontual", "<=", "<", ">=", ">" ou "entre".
    a, b : int
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = geom(p=p)

    valor = _prob_discreta(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Geométrica",
        parametros={"p": p},
        x=x, tipo=tipo, valor=valor,
        media_teor=1 / p, var_teor=(1 - p) / p ** 2,
        a=a, b=b
    )


# distribuição hipergeométrica
def dist_hipergeometrica(x=None, N_pop=20, K=7, n=5, tipo="pontual", a=None, b=None):
    """
    Distribuição Hipergeométrica.

    Parameters
    ----------
    x : int
        Número de sucessos observados na amostra.
    N_pop : int
        Tamanho da população.
    K : int
        Número de sucessos existentes na população.
    n : int
        Tamanho da amostra retirada (sem reposição).
    tipo : str
        "pontual", "<=", "<", ">=", ">" ou "entre".
    a, b : int
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = hypergeom(M=N_pop, n=K, N=n)

    valor = _prob_discreta(dist, x, tipo, a, b)

    media_teor = n * (K / N_pop)

    var_teor = (
        media_teor
        * ((N_pop - K) / N_pop)
        * ((N_pop - n) / (N_pop - 1))
    )

    return _montar_resultado(
        nome_dist="Hipergeométrica",
        parametros={"N": N_pop, "K": K, "n": n},
        x=x, tipo=tipo, valor=valor,
        media_teor=media_teor, var_teor=var_teor,
        a=a, b=b
    )


# distribuição binomial negativa
def dist_binomial_negativa(x=None, r=1, p=0.5, tipo="pontual", a=None, b=None):
    """
    Distribuição Binomial Negativa.

    Modela o número de falhas observadas antes de atingir
    'r' sucessos, cada tentativa com probabilidade 'p' de sucesso.

    Parameters
    ----------
    x : int
        Número de falhas observadas.
    r : int
        Número de sucessos desejados.
    p : float
        Probabilidade de sucesso em cada tentativa.
    tipo : str
        "pontual", "<=", "<", ">=", ">" ou "entre".
    a, b : int
        Limites usados quando tipo="entre".

    Returns
    -------
    dict
    """

    dist = nbinom(n=r, p=p)

    valor = _prob_discreta(dist, x, tipo, a, b)

    return _montar_resultado(
        nome_dist="Binomial Negativa",
        parametros={"r": r, "p": p},
        x=x, tipo=tipo, valor=valor,
        media_teor=r * (1 - p) / p, var_teor=r * (1 - p) / p ** 2,
        a=a, b=b
    )




#############################################
#        DISTRIBUIÇÕES MULTIVARIADAS        #
#############################################


# distribuição multinomial
def dist_multinomial(x, n, p):
    """
    Distribuição Multinomial.

    Parameters
    ----------
    x : array-like
        Vetor de contagens observadas (soma deve ser igual a n).
    n : int
        Número total de tentativas.
    p : array-like
        Vetor de probabilidades de cada categoria (soma = 1).

    Returns
    -------
    dict
    """

    x = np.asarray(x)
    p = np.asarray(p)

    valor = multinomial.pmf(x, n=n, p=p)

    media_teor = (n * p).tolist()
    var_teor = (n * p * (1 - p)).tolist()

    return {
        "distribuicao": "Multinomial",
        "parametros": {"n": n, "p": p.tolist()},
        "x": x.tolist(),
        "probabilidade": float(valor),
        "media_teorica": media_teor,
        "variancia_teorica": var_teor,
        "interpretacao": (
            f"A probabilidade de observar o vetor de contagens {x.tolist()} "
            f"em {n} tentativas, segundo a distribuição Multinomial, "
            f"é igual a {valor:.4f} ({valor * 100:.2f}%)."
        )
    }


# distribuição normal multivariada
def dist_normal_multivariada(x, media, cov):
    """
    Distribuição Normal Multivariada.

    Parameters
    ----------
    x : array-like
        Ponto onde a densidade será avaliada.
    media : array-like
        Vetor de médias.
    cov : array-like
        Matriz de covariância.

    Returns
    -------
    dict
    """

    x = np.asarray(x)
    media = np.asarray(media)
    cov = np.asarray(cov)

    dist = multivariate_normal(mean=media, cov=cov)

    valor = dist.pdf(x)

    return {
        "distribuicao": "Normal Multivariada",
        "parametros": {"media": media.tolist(), "cov": cov.tolist()},
        "x": x.tolist(),
        "densidade": float(valor),
        "interpretacao": (
            f"A densidade de probabilidade conjunta no ponto {x.tolist()}, "
            f"segundo a Normal Multivariada, é igual a {valor:.6f}."
        )
    }


# distribuição dirichlet
def dist_dirichlet(x, alfa):
    """
    Distribuição Dirichlet.

    Parameters
    ----------
    x : array-like
        Vetor de proporções (soma = 1), ponto de avaliação da densidade.
    alfa : array-like
        Vetor de parâmetros de concentração.

    Returns
    -------
    dict
    """

    x = np.asarray(x)
    alfa = np.asarray(alfa)

    dist = dirichlet(alpha=alfa)

    valor = dist.pdf(x)

    soma_alfa = alfa.sum()
    media_teor = (alfa / soma_alfa).tolist()

    return {
        "distribuicao": "Dirichlet",
        "parametros": {"alfa": alfa.tolist()},
        "x": x.tolist(),
        "densidade": float(valor),
        "media_teorica": media_teor,
        "interpretacao": (
            f"A densidade de probabilidade no ponto {x.tolist()}, "
            f"segundo a distribuição Dirichlet, é igual a {valor:.6f}."
        )
    }




