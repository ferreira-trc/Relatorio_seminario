import pandas as pd
import numpy as np
from docplex.mp.model import Model

# Leitura dos dados do arquivo Excel
dados_covariancia = pd.read_excel('Otimização.xlsx', sheet_name='Covariancia', index_col=0)

# Transformação da matriz de covariância em lista bidimensional
covariancia_lista = dados_covariancia.values.tolist()

# Preenchimento dos valores ausentes com os valores espelhados
covariancia_simetrica = np.array(covariancia_lista)
covariancia_simetrica[np.isnan(covariancia_simetrica)] = covariancia_simetrica.T[np.isnan(covariancia_simetrica)]


# Leitura dos dados do arquivo Excel
dados_retorno = pd.read_excel('Otimização.xlsx', sheet_name='Retornos', index_col=0)

# Extração da coluna desejada como lista
coluna_lista = dados_retorno['Medias'].tolist()
coluna_tickers = dados_retorno['Ticker'].tolist()


# Criação do modelo
modelo = Model(name='Otimização de Carteira')

# Número de ativos no portfólio
num_ativos = len(coluna_lista)

# Variáveis de decisão
tickers = {coluna_tickers[i]: modelo.continuous_var(lb=0, ub=1, name=coluna_tickers[i]) for i in range(num_ativos)}

#funções 
# Retorno esperado do portfólio
retorno_esperado = modelo.sum(tickers[i] * coluna_lista[coluna_tickers.index(i)] for i in tickers)
# Risco do portfólio (variância)
risco_portfolio = modelo.sum(
    modelo.sum(tickers[i] * tickers[j] * covariancia_simetrica[coluna_tickers.index(i)][coluna_tickers.index(j)] for j in coluna_tickers)
    for i in coluna_tickers
)

risco_portfolio_min = 0.0002

# Restrição: 
modelo.add_constraint(modelo.sum(tickers[i] for i in tickers) == 1)
modelo.add_constraint(risco_portfolio <= risco_portfolio_min)


# Problema de otimização: Maxmizar o retorno do portfólio sujeito a um risco maximo
modelo.maximize(retorno_esperado)

# Resolução do modelo
solucao = modelo.solve()
    
print(modelo.export_to_string())

solucion = modelo.solve(log_output=True)
print(modelo.get_solve_status())

solucion.display() 
