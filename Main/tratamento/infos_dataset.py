
import pandas as pd

df = pd.read_csv(r'..\dataset\br_inpe_prodes_municipio_bioma.csv')

print(df.columns.to_list())
print(df.dtypes)
print("Número de registros:", len(df))
print("Número de municípios:", df['id_municipio'].nunique())
print(df.head())
print(df.describe())
print(df['bioma'].value_counts())
print(df.isnull().sum())

negativos = df[df['vegetacao_natural'] < 0]
print("Número de registros com vegetação natural negativa:", len(negativos))

# 347 registros com valor negativo para vegetação natural de um total de 150k+ ent vou mudar para zero
df['vegetacao_natural'] = df['vegetacao_natural'].clip(lower=0)
print("Registros com vegetação natural negativa após correção:",
      len(df[df['vegetacao_natural'] < 0]))

# verificando inconsistências entre colunas de área
df['soma_partes'] = df['vegetacao_natural'] + df['nao_vegetacao_natural'] + df['hidrografia']
inconsistentes = df[df['soma_partes'] > df['area_total']]
print("Número de registros inconsistentes:", len(inconsistentes))
print(inconsistentes.head())
# deu 466 registros inconsistentes mas a diferença é muito pouca, provavelmente erro de arredondamento (23.2 e 23.18354)
# ent por enquanto o dataset tá legal. Vou salvar o arquivo corrigido
df.to_csv("..\dataset\dataset_corrigido.csv", index=False)