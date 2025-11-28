from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def load_and_prepare():
    base = Path(__file__).resolve().parents[1]
    path = base / 'dataset' / 'dataset_corrigido.csv'
    df = pd.read_csv(path)
    df = df.sort_values(['id_municipio', 'ano'])
    df['desmatado_prev'] = df.groupby('id_municipio')['desmatado'].shift(1).fillna(0)
    features = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural',
                'hidrografia', 'desmatado_prev', 'bioma']
    target = 'desmatado'
    df = df.dropna(subset=features + [target])
    X = df[features].copy()
    y = df[target].copy()
    return X, y, df


def build_preproc(numeric_cols, categorical_cols):
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, categorical_cols)])
    return preproc


def main():
    X, y, df = load_and_prepare()
    train_idx = df['ano'] <= 2018
    test_idx = df['ano'] > 2018
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print('Tamanhos: treino', len(X_train), 'teste', len(X_test))
    print('\nResumo y_test:')
    print(y_test.describe())

    numeric_cols = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia', 'desmatado_prev']
    categorical_cols = ['bioma']
    preproc = build_preproc(numeric_cols, categorical_cols)

    lr = Pipeline([('preproc', preproc), ('model', LinearRegression())])
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f'LR RMSE: {rmse:.6f}, R2: {r2:.6f}')

    # inspecionar correlação com desmatado_prev
    print('\nCorrelação entre desmatado_prev e desmatado (teste):')
    merged = X_test.copy()
    merged['desmatado'] = y_test.values
    print(merged[['desmatado_prev', 'desmatado']].corr())

    # verificar diferenças entre predição e real
    dif = pd.Series(pred, index=y_test.index) - y_test
    print('\nErro absoluto médio (teste):', dif.abs().mean())
    print('Número de previsões idênticas ao real:', (dif == 0).sum())
    print('\nExemplos onde predição != real (primeiros 10):')
    neq = dif[ dif.abs() > 1e-8 ]
    print(neq.head(10))

    # checar se alguma coluna é idêntica ao target
    print('\nVerificando colunas idênticas ao target (amostras do teste):')
    for col in X_test.columns:
        # skip categorical by conversion
        if X_test[col].dtype == 'object':
            continue
        try:
            if np.allclose(X_test[col].values.astype(float), y_test.values.astype(float)):
                print('Coluna idêntica ao target:', col)
        except Exception:
            pass

    # mostrar primeiras linhas do test set com preds
    out = X_test.copy()
    out['desmatado_real'] = y_test.values
    out['pred_lr'] = pred
    print('\nAmostra do test set com predição:')
    print(out.head(10))


if __name__ == '__main__':
    main()
