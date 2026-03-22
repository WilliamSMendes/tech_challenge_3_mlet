# Tech Challenge 3 — Previsão de Atraso de Voos

Projeto desenvolvido como parte do **Tech Challenge 3** do curso de **Machine Learning Engineering** da FIAP. O objetivo é analisar dados de voos dos Estados Unidos para compreender os padrões de atraso, construir modelos preditivos de classificação e segmentar voos por meio de clustering.

## Dados

Os dados utilizados neste projeto devem ser obtidos no seguinte link e adicionados à pasta `dados/`:

🔗 [Google Drive — Dados do Projeto](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj)

Os arquivos esperados são `airlines.csv`, `airports.csv` e `flights.csv`.

## Instalação

```bash
pip install -r requirements.txt
```

## Estrutura dos Notebooks

### 01_eda.ipynb — Análise Exploratória (Visão de Negócio)

Integração dos três datasets (airlines, airports e flights) e análise exploratória sob perspectiva operacional. São investigados padrões de atraso por companhia aérea, dia da semana e horário do dia, além da distribuição estatística dos tempos de voo e atrasos. Identificam-se as principais causas de atraso (Late Aircraft Delay, Air System Delay, Airline Delay) e observa-se que mais de 1 em cada 3 voos apresenta atraso, com segunda-feira e o horário das 20h como os períodos mais críticos. O dataset integrado é salvo em `dados/df_final.csv`.

### 02_eda.ipynb — Análise Exploratória (Visão Técnica)

Avalia a qualidade dos dados com foco em tipagem de variáveis, valores ausentes, outliers e correlações. A análise de missing values identifica colunas com mais de 80% de nulos (atrasos climáticos, motivo de cancelamento), além de variáveis com ~8% e 1-2% de ausência. Aplica-se detecção de outliers via método IQR e analisa-se a correlação de Pearson entre variáveis numéricas. São criadas as variáveis-alvo binárias (ATRASO_SAIDA e ATRASO_CHEGADA, com limiar de 10 minutos) e define-se a estratégia de tratamento para a etapa de feature engineering.

### 03_feature_eng.ipynb — Feature Engineering e Pré-processamento

Preparam os dados para modelagem por meio de filtragem de voos cancelados e desviados, remoção de colunas com data leakage, imputação de valores ausentes (mediana para numéricos, moda para categóricos) e capping de outliers via percentis p1/p99. Categorias raras são agrupadas sob o rótulo "OUTROS". São criadas novas features de domínio: período do dia, tipo de dia da semana, estação do ano, faixa de distância, horário de pico, codificação cíclica (seno/cosseno) para variáveis temporais e métricas de congestionamento aeroportuário (volume de voos por aeroporto/hora). Os conjuntos de treino e teste são exportados para `dados/`.

### 04_classification.ipynb — Modelagem de Classificação

Treina três modelos de classificação (XGBoost, LightGBM e Random Forest) para prever atrasos de partida. O desbalanceamento de classes é tratado com SMOTE. A avaliação utiliza validação cruzada estratificada com 5 folds e métricas de Accuracy, Precision, Recall, F1-Score e ROC-AUC. O Random Forest é selecionado como melhor modelo por apresentar recall superior (~0.48), melhor F1-Score (~0.46) e melhor ROC-AUC (~0.72). São geradas análises de Feature Importance (Gini) e SHAP values para interpretabilidade. Em seguida, aplica-se fine-tuning com RandomizedSearchCV para otimização de hiperparâmetros.

### 05_clustering.ipynb — Segmentação por Clustering

Aplica o algoritmo KMeans para segmentar voos em grupos com comportamento operacional distinto. As features utilizadas incluem variáveis temporais (mês, dia da semana, hora), operacionais (atrasos, distância) e geográficas (coordenadas). A seleção do número ótimo de clusters é feita via Método do Cotovelo e Silhouette Score. Cada cluster é caracterizado com perfis estatísticos, radar charts e heatmaps comparativos, resultando em personas operacionais que possibilitam intervenções direcionadas por segmento de voo.

## Autores

- William Mendes
- André Beraldo
- Guilherme dos Santos
