from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle


def load_data():
    base = Path(__file__).resolve().parents[1]
    path = base / 'dataset' / 'dataset_corrigido.csv'
    df = pd.read_csv(path)
    return df


def prepare_features(df):
    # ordenar e criar variável com desmatamento do ano anterior por município
    df = df.sort_values(['id_municipio', 'ano'])
    df['desmatado_prev'] = df.groupby('id_municipio')['desmatado'].shift(1)

    # preencher NaNs de desmatado_prev com 0 (municipios sem ano anterior)
    df['desmatado_prev'] = df['desmatado_prev'].fillna(0)

    # escolher features simples e alvo
    features = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural',
                'hidrografia', 'desmatado_prev', 'bioma']
    target = 'desmatado'
    df = df.dropna(subset=features + [target])
    X = df[features].copy()
    y = df[target].copy()
    return X, y, df


def build_pipeline(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])

    return preproc


def train_and_evaluate(X_train, X_test, y_train, y_test, preproc):
    numeric_cols = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia', 'desmatado_prev']
    categorical_cols = ['bioma']

    # Linear Regression
    lr = Pipeline([
        ('preproc', preproc),
        ('model', LinearRegression())
    ])
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # Decision Tree
    dt = Pipeline([
        ('preproc', preproc),
        ('model', DecisionTreeRegressor(max_depth=6, random_state=42))
    ])
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)

    results = {
        'lr_rmse': np.sqrt(mean_squared_error(y_test, pred_lr)),
        'lr_r2': r2_score(y_test, pred_lr),
        'dt_rmse': np.sqrt(mean_squared_error(y_test, pred_dt)),
        'dt_r2': r2_score(y_test, pred_dt)
    }

    return lr, dt, results, pred_lr, pred_dt


def save_models(models, base):
    out = base / 'outputs'
    out.mkdir(exist_ok=True)
    for name, model in models.items():
        with open(out / f'{name}.pkl', 'wb') as f:
            pickle.dump(model, f)


def main():
    base = Path(__file__).resolve().parents[1]
    df = load_data()
    X, y, df_full = prepare_features(df)

    # separar treino/test por ano para simular predição temporal
    train_idx = df_full['ano'] <= 2018
    test_idx = df_full['ano'] > 2018

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    numeric_cols = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia', 'desmatado_prev']
    categorical_cols = ['bioma']
    preproc = build_pipeline(numeric_cols, categorical_cols)

    lr, dt, results, pred_lr, pred_dt = train_and_evaluate(X_train, X_test, y_train, y_test, preproc)

    print('Resultados (avaliação em dados de teste):')
    print(f"LinearRegression RMSE: {results['lr_rmse']:.3f}, R2: {results['lr_r2']:.3f}")
    print(f"DecisionTree RMSE: {results['dt_rmse']:.3f}, R2: {results['dt_r2']:.3f}")

    # salvar modelos
    save_models({'linear_regression': lr, 'decision_tree': dt}, base)

    # plot real x predito para árvore de decisão
    out = base / 'outputs'
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=pred_dt, alpha=0.3)
    plt.xlabel('Desmatado real')
    plt.ylabel('Desmatado predito (DT)')
    plt.title('Real vs Predito - Decision Tree')
    plt.tight_layout()
    plt.savefig(out / 'real_vs_predito_dt.png')
    print(f"Gráficos e modelos salvos em: {out}\n")


if __name__ == '__main__':
    main()
