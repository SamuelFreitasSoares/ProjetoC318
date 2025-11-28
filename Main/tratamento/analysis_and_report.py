from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def load_data():
    base = Path(__file__).resolve().parents[1]
    path = base / 'dataset' / 'dataset_corrigido.csv'
    df = pd.read_csv(path)
    return df, base


def prepare(df, use_prev=True):
    df = df.sort_values(['id_municipio', 'ano']).copy()
    if use_prev:
        df['desmatado_prev'] = df.groupby('id_municipio')['desmatado'].shift(1).fillna(0)
    features = ['ano', 'area_total', 'vegetacao_natural', 'nao_vegetacao_natural', 'hidrografia']
    if use_prev:
        features = features + ['desmatado_prev']
    features = features + ['bioma']
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


def fit_and_eval(X_train, X_test, y_train, y_test, preproc, model):
    pipe = Pipeline([('preproc', preproc), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return pipe, pred, rmse, r2


def time_series_cv(X, y, preproc, model, n_splits=5):
    pipe = Pipeline([('preproc', preproc), ('model', model)])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores


def get_feature_names(preproc, numeric_cols, categorical_cols):
    try:
        names = list(preproc.get_feature_names_out())
        return names
    except Exception:
        pass

    names = []
    for name, transformer, cols in preproc.transformers:
        if name == 'num':
            names.extend(cols)
        elif name == 'cat':
            ohe = transformer.named_steps['ohe']
            cats_attr = getattr(ohe, 'categories_', None)
            if cats_attr is None:
                cats_attr = getattr(ohe, 'categories', None)
            if cats_attr is None:
                names.extend(cols)
            else:
                for cat, cats in zip(cols, cats_attr):
                    names.extend([f"{cat}__{c}" for c in cats])
    return names


def save_plot(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    df, base = load_data()

    results = {}
    out = base / 'outputs'
    out.mkdir(exist_ok=True)

    for use_prev in [True, False]:
        label = 'with_prev' if use_prev else 'no_prev'
        X, y, df_p = prepare(df, use_prev=use_prev)

        # split temporal
        train_idx = df_p['ano'] <= 2018
        test_idx = df_p['ano'] > 2018
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        numeric_cols = [c for c in X.columns if c != 'bioma']
        categorical_cols = ['bioma']
        preproc = build_preproc(numeric_cols, categorical_cols)

        preproc.fit(X_train)

        lr_model = LinearRegression()
        lr_pipe, lr_pred, lr_rmse, lr_r2 = fit_and_eval(X_train, X_test, y_train, y_test, preproc, lr_model)

        dt_model = DecisionTreeRegressor(max_depth=6, random_state=42)
        dt_pipe, dt_pred, dt_rmse, dt_r2 = fit_and_eval(X_train, X_test, y_train, y_test, preproc, dt_model)

        tscv_lr = time_series_cv(X, y, preproc, lr_model, n_splits=5)
        tscv_dt = time_series_cv(X, y, preproc, dt_model, n_splits=5)

        with open(out / f'lr_{label}.pkl', 'wb') as f:
            pickle.dump(lr_pipe, f)
        with open(out / f'dt_{label}.pkl', 'wb') as f:
            pickle.dump(dt_pipe, f)

        try:
            feat_names = list(preproc.get_feature_names_out())
        except Exception:
            feat_names = get_feature_names(preproc, numeric_cols, categorical_cols)

        lr_coef = lr_pipe.named_steps['model'].coef_
        if len(lr_coef) == len(feat_names):
            coef_series = pd.Series(lr_coef, index=feat_names)
        else:
            coef_series = pd.Series(lr_coef)
        coef_series = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)

        dt_imp = dt_pipe.named_steps['model'].feature_importances_
        if len(dt_imp) == len(feat_names):
            imp_series = pd.Series(dt_imp, index=feat_names)
        else:
            imp_series = pd.Series(dt_imp)
        imp_series = imp_series.sort_values(ascending=False)

        fig1 = plt.figure(figsize=(8,6))
        coef_series.head(20).plot(kind='bar')
        plt.title(f'LR top coefficients ({label})')
        save_plot(fig1, out / f'lr_coefs_{label}.png')

        fig2 = plt.figure(figsize=(8,6))
        imp_series.head(20).plot(kind='bar', color='C1')
        plt.title(f'DT top importances ({label})')
        save_plot(fig2, out / f'dt_imps_{label}.png')

        fig3 = plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_test, y=dt_pred, alpha=0.3)
        plt.xlabel('real')
        plt.ylabel('pred_dt')
        plt.title(f'Real vs Pred DT ({label})')
        save_plot(fig3, out / f'real_vs_pred_dt_{label}.png')
        results[label] = {
            'lr_rmse': float(lr_rmse),
            'lr_r2': float(lr_r2),
            'dt_rmse': float(dt_rmse),
            'dt_r2': float(dt_r2),
            'tscv_lr_rmse_mean': float(np.mean(tscv_lr)),
            'tscv_lr_rmse_std': float(np.std(tscv_lr)),
            'tscv_dt_rmse_mean': float(np.mean(tscv_dt)),
            'tscv_dt_rmse_std': float(np.std(tscv_dt)),
            'coef_top': coef_series.head(20).to_dict(),
            'imp_top': imp_series.head(20).to_dict()
        }

    docs = base / 'docs'
    docs.mkdir(exist_ok=True)
    md = docs / 'relatorio.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Relatório - Análise e Modelagem (PRODES)\n\n')
        f.write('Este relatório resume os resultados dos experimentos feitos com e sem a feature `desmatado_prev`.\n\n')
        for k, v in results.items():
            f.write(f'## Experimento: {k}\n')
            f.write(f"- LinearRegression RMSE: {v['lr_rmse']:.3f}, R2: {v['lr_r2']:.3f}\n")
            f.write(f"- DecisionTree RMSE: {v['dt_rmse']:.3f}, R2: {v['dt_r2']:.3f}\n")
            f.write(f"- TimeSeriesSplit LR RMSE (mean ± std): {v['tscv_lr_rmse_mean']:.3f} ± {v['tscv_lr_rmse_std']:.3f}\n")
            f.write(f"- TimeSeriesSplit DT RMSE (mean ± std): {v['tscv_dt_rmse_mean']:.3f} ± {v['tscv_dt_rmse_std']:.3f}\n")
            f.write('\nTop coeficientes (LR):\n')
            for name, val in v['coef_top'].items():
                f.write(f'- {name}: {val:.6f}\n')
            f.write('\nTop importâncias (DT):\n')
            for name, val in v['imp_top'].items():
                f.write(f'- {name}: {val:.6f}\n')
            f.write('\nImagens (outputs/):\n')
            f.write(f"![LR Coefs](../outputs/lr_coefs_{k}.png)\n")
            f.write(f"![DT Imps](../outputs/dt_imps_{k}.png)\n")
            f.write(f"![Real x Pred DT](../outputs/real_vs_pred_dt_{k}.png)\n")
            f.write('\n---\n')

        f.write('\n\n**Interpretação breve**:\n')
        f.write('- A inclusão de `desmatado_prev` tende a reduzir fortemente o erro (efeito autoregressivo).\n')
        f.write('- Recomenda-se avaliar features exógenas para previsão útil de prevenção.\n')

    print('Análises executadas. Relatório salvo em', md)


if __name__ == '__main__':
    main()
