import pandas as pd
import numpy as np
from docplex.mp.model import Model

# Leitura dos dados do arquivo Excel
dados_covariancia = pd.read_excel('Otimização.xlsx', sheet_name='Covariancia', index_col=0)
dados_retorno = pd.read_excel('Otimização.xlsx', sheet_name='Folha5', index_col=0)
dados_paises = pd.read_excel('Analise de Carteira.xlsx', sheet_name='Ativos_Nacionalidade_Moeda', index_col=0)

# Transformação dos dados em estruturas adequadas
covariancia_lista = dados_covariancia.values.tolist()
coluna_lista = dados_retorno['Medias'].tolist()
coluna_tickers = dados_retorno['Ticker'].tolist()
coluna_paises = dados_paises['País'].tolist()



# Preenchimento dos valores ausentes com os valores espelhados
covariancia_simetrica = np.array(covariancia_lista)
covariancia_simetrica[np.isnan(covariancia_simetrica)] = covariancia_simetrica.T[np.isnan(covariancia_simetrica)]

# Criação do modelo
modelo = Model(name='Otimização de Carteira')

# Número de ativos no portfólio
num_ativos = len(coluna_lista)

# Variáveis de decisão
tickers = {coluna_tickers[i]: modelo.continuous_var(lb=0, ub=1, name=coluna_tickers[i]) for i in range(num_ativos)}

# Funções
retorno_esperado = modelo.sum(tickers[i] * coluna_lista[coluna_tickers.index(i)] for i in tickers)
risco_portfolio = modelo.sum(
    modelo.sum(tickers[i] * tickers[j] * covariancia_simetrica[coluna_tickers.index(i)][coluna_tickers.index(j)] for j in coluna_tickers)
    for i in coluna_tickers
)
risco_portfolio_min = 0.0002

# Restrição: Soma dos pesos dos ativos deve ser igual a 1
modelo.add_constraint(modelo.sum(tickers[i] for i in tickers) == 1)

# Dicionário para armazenar os pesos dos países
pesos_paises = {}

# Agrupamento dos ativos por país
for pais in coluna_paises:
    ativos_pais = [ativo for ativo, p in zip(coluna_tickers, coluna_paises) if p == pais]
    pesos_paises[pais] = modelo.sum(tickers[ativo] for ativo in ativos_pais)

# Restrição: Soma dos pesos dos países deve ser menor ou igual a uma constante específica
for pais, peso_maximo in zip(pesos_paises.keys(), peso_maximo_por_pais):
    modelo.add_constraint(pesos_paises[pais] <= peso_maximo)
    
    
# Restrição: Risco do portfólio deve ser menor ou igual a um limite
modelo.add_constraint(risco_portfolio <= risco_portfolio_min)

# Problema de otimização: Maximizar o retorno do portfólio
modelo.maximize(retorno_esperado)

# Resolução do modelo
solucao = modelo.solve()

# Impressão da solução
print(modelo.export_to_string())
solucion = modelo.solve(log_output=True)
print(modelo.get_solve_status())
solucion.display()
