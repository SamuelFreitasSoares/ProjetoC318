# Projeto: Análise de Desmatamento no Brasil (PRODES)

## Descrição
Este projeto tem como objetivo analisar o **desmatamento no Brasil** utilizando dados do **PRODES**, que monitora o desmatamento por corte raso na Amazônia Legal desde 1988. O projeto segue as etapas clássicas de um workflow de ciência de dados e Machine Learning, fornecendo insights sobre padrões temporais e espaciais do desmatamento e da vegetação natural.

---

## Etapas do Projeto

### 1. Contextualização e definição do problema
- Entender os objetivos do projeto, tanto do ponto de vista ambiental quanto de análise de dados.  
- Formular perguntas de negócio que possam ser respondidas com o dataset.

### 2. Coleta e limpeza de dados
- Importação dos dados por município e ano.  
- Correção de inconsistências, como valores negativos de vegetação natural.  
- Verificação de somas das áreas e identificação de outliers.

### 3. Análise Exploratória de Dados (AED)
- Estatísticas descritivas e distribuição das variáveis.  
- Histogramas e boxplots de desmatamento, vegetação natural e hidrografia.  
- Cálculo de percentuais de vegetação natural, não vegetação e hidrografia.  
- Evolução do desmatamento e da vegetação ao longo dos anos.  
- Comparação entre biomas.

### 4. Preparação para modelagem
- Identificação de variáveis relevantes para análise e modelagem.  
- Transformações e engenharia de features para futuros modelos de Machine Learning.

### 5. Modelagem e avaliação
- Criação de modelos preditivos ou de classificação conforme os objetivos do projeto.  
- Avaliação da performance e interpretação dos resultados.

### 6. Geração de resultados e apresentação
- Visualização de insights obtidos na AED e na modelagem.  
- Comunicação clara dos resultados
---

## Objetivo final
O projeto visa fornecer **uma visão abrangente do desmatamento no Brasil**, servindo como base para análises ambientais detalhadas e para futuras etapas de Machine Learning.

**Contexto e objetivos**: O objetivo deste projeto é construir uma análise e um pipeline simples de Machine Learning capaz de prever o volume de desmatamento (`desmatado`) em municípios brasileiros com base em dados do PRODES. Do ponto de vista de negócio, a previsão pode ser usada para identificar áreas com maior risco e priorizar ações de prevenção e fiscalização. Do ponto de vista de ciência de dados, o objetivo é praticar o fluxo completo: ingestão, limpeza, análise exploratória, engenharia de features, modelagem e avaliação.

**Enquadramento do problema**: Para manter o projeto no nível iniciante, optamos por um problema de Aprendizagem Supervisionada — Regressão. A tarefa é prever a área desmatada em um município/ano com modelos simples (Regressão Linear e Decision Tree) usando features como área total, vegetação natural, hidrografia, bioma e desmatamento do ano anterior.

**Como executar**:
- **Instalar dependências**: com o Python ativo no ambiente (recomenda-se venv), execute:

```powershell
python -m pip install -r requirements.txt
```

- **Rodar o script de tratamento inicial** (opcional, já existe `dataset/dataset_corrigido.csv`):

```powershell
py .\tratamento\infos_dataset.py
```

- **Rodar o pipeline básico (EDA + modelagem)**:

```powershell
py .\tratamento\pipeline_basico.py
```

Os modelos treinados e os gráficos serão salvos em `outputs/` dentro da pasta `Main`.
