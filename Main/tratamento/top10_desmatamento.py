from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def fetch_municipio_names(codes):
    """Consulta a API do IBGE para obter nomes dos municípios a partir dos códigos."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(500, 502, 504))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    mapping = {}
    for code in sorted(set(codes)):
        try:
            url = f'https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{int(code)}'
            r = session.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                mapping[int(code)] = data.get('nome')
            else:
                mapping[int(code)] = None
        except Exception:
            mapping[int(code)] = None
    return mapping


def main():
    base = Path(__file__).resolve().parents[1]
    path = base / 'dataset' / 'dataset_corrigido.csv'
    df = pd.read_csv(path)

    # criar variável com desmatamento do ano anterior por município
    df = df.sort_values(['id_municipio', 'ano'])
    df['desmatado_prev'] = df.groupby('id_municipio')['desmatado'].shift(1).fillna(0)

    # features e alvo (mesma escolha do pipeline básico)
    features = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural',
                'hidrografia', 'desmatado_prev', 'bioma']
    target = 'desmatado'

    df = df.dropna(subset=features + [target])

    X = df[features].copy()
    y = df[target].copy()

    # separar treino/test por ano (<=2018 treina, >2018 testa)
    train_idx = df['ano'] <= 2018
    test_idx = df['ano'] > 2018

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    numeric_cols = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia', 'desmatado_prev']
    categorical_cols = ['bioma']

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])

    # Decision Tree
    model_dt = Pipeline([
        ('preproc', preproc),
        ('model', DecisionTreeRegressor(max_depth=6, random_state=42))
    ])
    model_dt.fit(X_train, y_train)
    preds_dt = model_dt.predict(X_test)

    # Linear Regression (para comparar rankings)
    model_lr = Pipeline([
        ('preproc', preproc),
        ('model', LinearRegression())
    ])
    model_lr.fit(X_train, y_train)
    preds_lr = model_lr.predict(X_test)

    # anexar predição ao dataframe original (apenas para os índices de teste)
    df.loc[test_idx, 'desmatado_pred_dt'] = preds_dt
    df.loc[test_idx, 'desmatado_pred_lr'] = preds_lr

    # Top 10 observado (por id_municipio + ano)
    observed = (
        df.groupby(['id_municipio', 'ano'], as_index=False)
        .agg({'desmatado': 'sum', 'bioma': 'first', 'area_total': 'first'})
        .sort_values('desmatado', ascending=False)
    )

    top10_observed = observed.head(10)

    # Top 10 predito (apenas linhas de teste porque preditas)
    predicted = df.loc[test_idx, ['id_municipio', 'ano', 'desmatado_pred_dt', 'desmatado_pred_lr', 'desmatado', 'bioma', 'area_total']].copy()
    predicted_dt = predicted.sort_values('desmatado_pred_dt', ascending=False).head(10)
    predicted_lr = predicted.sort_values('desmatado_pred_lr', ascending=False).head(10)

    out = base / 'outputs'
    out.mkdir(exist_ok=True)

    top10_observed.to_csv(out / 'top10_observado.csv', index=False)
    predicted_dt.to_csv(out / 'top10_predito_dt.csv', index=False)
    predicted_lr.to_csv(out / 'top10_predito_lr.csv', index=False)

    print('\nTop 10 — Desmatamento observado (maior para menor):')
    print(top10_observed.to_string(index=False))

    print('\nTop 10 — Desmatamento predito (Decision Tree) [apenas conjunto de teste]:')
    print(predicted_dt.to_string(index=False))

    print('\nTop 10 — Desmatamento predito (Linear Regression) [apenas conjunto de teste]:')
    print(predicted_lr.to_string(index=False))

    # mapear nomes via API do IBGE para os top10
    codes_to_map = list(top10_observed['id_municipio'].head(10)) + list(predicted_dt['id_municipio'].head(10)) + list(predicted_lr['id_municipio'].head(10))
    mapping = fetch_municipio_names(codes_to_map)

    # anexar nomes aos dataframes
    top10_observed['municipio_nome'] = top10_observed['id_municipio'].map(mapping)
    predicted_dt['municipio_nome'] = predicted_dt['id_municipio'].map(mapping)
    predicted_lr['municipio_nome'] = predicted_lr['id_municipio'].map(mapping)

    # salvar versões com nomes
    top10_observed.to_csv(out / 'top10_observado_com_nomes.csv', index=False)
    predicted_dt.to_csv(out / 'top10_predito_dt_com_nomes.csv', index=False)
    predicted_lr.to_csv(out / 'top10_predito_lr_com_nomes.csv', index=False)

    print('\nArquivos com nomes salvos em:', out)


if __name__ == '__main__':
    main()
