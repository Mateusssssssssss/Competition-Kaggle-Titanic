# Titanic - Predição de Sobreviventes

Este projeto tem como objetivo prever a sobrevivência de passageiros do Titanic com base em características como idade, classe da cabine, tarifa paga e gênero. Utilizamos a base de dados do Kaggle e aplicamos técnicas de aprendizado de máquina para realizar a classificação.

## Tecnologias Utilizadas
- **Pandas**: Manipulação e análise de dados
- **Seaborn & Matplotlib**: Visualização de dados
- **Scikit-Learn**:
  - `LabelEncoder`: Codifica variáveis categóricas
  - `train_test_split`: Divide os dados em treinamento e teste
  - `ExtraTreesClassifier`: Modelo de aprendizado de máquina para classificação
  - `confusion_matrix`, `accuracy_score`: Métricas de avaliação
  - `cross_val_score`: Validação cruzada

## Estrutura do Projeto
1. **Leitura e inspeção dos dados**
   - Carregamento do dataset
   - Exibição de estatísticas descritivas
   - Verificação de valores nulos
   
2. **Tratamento de Dados**
   - Substitui valores nulos na coluna `Age` pela mediana
   - Verifica e trata outliers
   - Codifica a coluna `Sex` para valores numéricos (0 e 1)
   
3. **Análise de Correlação**
   - Identifica relações entre variáveis importantes para previsão
   
4. **Divisão dos Dados**
   - Divide os dados entre conjunto de treinamento (70%) e teste (30%)
   
5. **Treinamento do Modelo**
   - Utiliza `ExtraTreesClassifier` para aprender padrões nos dados
   - Avalia a importância das variáveis no modelo
   
6. **Avaliação do Modelo**
   - Gera matriz de confusão para verificar acertos e erros
   - Calcula acurácia e taxa de erro
   - Aplica validação cruzada para garantir a confiabilidade do modelo
   
7. **Predição em Novos Dados**
   - Carrega o dataset de teste
   - Trata valores nulos
   - Aplica o modelo treinado para prever a sobrevivência
   - Salva as previsões em um novo arquivo CSV

## Resultados
- **Matriz de confusão**: Exibe a qualidade das previsões
- **Acurácia média**: Aproximadamente **80%**
- **Saída final**: Arquivo `teste_psclass_fare_age_sex_columns.csv` contendo as previsões

## Contribuição
Sinta-se à vontade para sugerir melhorias ou explorar outros modelos de machine learning para aprimorar as previsões!

