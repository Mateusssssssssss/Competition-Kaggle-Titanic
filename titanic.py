# Manipular os dados
import pandas as pd
# Divide o conjunto de dados em duas partes: uma para treinamento e outra para teste.
from sklearn.model_selection import train_test_split
#Converte rótulos categóricos (como texto) em valores numéricos
# para que possam ser usados em modelos de machine learning.
from sklearn.preprocessing import LabelEncoder
# confusion_matrix: Calcula a matriz de confusão, que mostra o desempenho de um classificador, 
# comparando as previsões com os valores reais.
#Avaliar o desempenho de modelos de classificação (ex.: True Positives, False Positives, etc.).
# acuracu_score: Calcula a acurácia do modelo, ou seja, a porcentagem de previsões corretas.
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#é utilizado para treinar um modelo de árvore de decisão. Ele é um classificador baseado em floresta aleatória (ensemble), 
# que pode calcular a importância das características (atributos) do conjunto de dados durante o treinamento.
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
#Para visualização
import seaborn as sb
#Para visualização
import matplotlib.pyplot as plt
#Validação cruzada 
from sklearn.model_selection import cross_val_score

# Ler o dataset
dados = pd.read_csv('train.csv')
print(dados.head())
print(dados.describe())
print(dados.shape)
# Quantidade de nulos
dados['Age'] = dados['Age'].fillna(dados['Age'].median())
null_quat = dados.isnull().sum()
print(null_quat)
#Verificação de outliers
sb.boxplot(dados[['Pclass', 'Fare']])
plt.show()
#Verificando media da coluna Age
age = dados['Age'].mean()
print(age)
#Verificando moda da coluna Age
age_moda = dados['Age'].mode()
print(age_moda)

#Frequencia com que a idade 24, aparece.
age_frequencia = (dados['Age'] == 24).sum()
print(age_frequencia)


# formato da matriz
previsores = dados.iloc[:,[2, 4, 5, 9]].values
sobreviventes = dados.iloc[:, 1].values


#Analise de correlação
correlacao = dados[['Survived', 'Age', 'Parch', 'Pclass', 'Parch','Fare','SibSp']].corr()
print(correlacao)


# Transformação dos atributos categóricos em atributos numéricos, 
# passando o índice de cada coluna categórica. Precisamos criar um objeto para cada atributo categórico, 
# pois na sequência vamos executar o processo de encoding novamente para o registro de teste
# Se forem utilizados objetos diferentes, o número atribuído a cada valor poderá ser diferente,
# o que deixará o teste inconsistente.
# Codificação de variáveis categóricas para variáveis numéricas.
labelencoder = LabelEncoder()
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])


# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
# random_state=1 é uma maneira de garantir consistência nos resultados, o que é útil quando você está tentando comparar modelos ou experimentos.
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 sobreviventes,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
print(x_teste)  



# Utilização do algoritimo ExtraTressClassifier para extrair as caracteristica mais importante
xg = XGBClassifier(objective='binary:logistic',  # Classificação binária
    eval_metric='aucpr',            # Métrica de avaliação
    n_estimators=600,             # Número de árvores
    learning_rate=0.02,           # Taxa de aprendizado
    max_depth=30,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.6,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=0,                 # Regularização L2
    reg_alpha=1,
    scale_pos_weight=10,             # dá mais peso para a classe minoritária
           )


xg.fit(x_treinamento, y_treinamento)
# Descobri qual os atributsos mais importantes
#extrai a importância de cada característica.
importancia = xg.feature_importances_
print(f'Importancia: {importancia}')


previsoes = xg.predict(x_teste)
print(previsoes)

#geração da matriz de confusão
#A matriz de confusão é uma ferramenta essencial para avaliar a performance de um modelo de classificação, 
# pois mostra não apenas os acertos do modelo (TP e TN), mas também os erros (FP e FN). 
# Isso ajuda a entender onde o modelo está errando e pode fornecer informações valiosas para ajustar o
# modelo ou o processo de treinamento
confusao = confusion_matrix(y_teste, previsoes)
print(f'Matriz de Confusão: {confusao}')

# calcula a taxa de acerto e a taxa de erro do modelo
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto

#Validação cruzada
#A validação cruzada é uma técnica que divide o conjunto de dados em várias partes (folds) e 
# treina o modelo em diferentes divisões, garantindo que o modelo tenha sido avaliado de maneira
# mais confiável e sem viés.
scores = cross_val_score(xg, x_treinamento, y_treinamento, cv=10)
print(f'Média da acurácia: {scores.mean()}')
# Média da acurácia: 0.8025601638504863
#Taxa de acerto: 0.8022388059701493
#Taxa de erro:0.1977611940298507
print(f'Taxa de acerto: {taxa_acerto}\nTaxa de erro:{taxa_erro}')

#Dataset de teste onde não possui a Coluna Survived
dados2 = pd.read_csv('test.csv')
print(dados.shape)
#Boxplot
sb.boxplot(dados2)
plt.title('Dados 2')
plt.show()
#Verificando valores nulos
print(dados2.isnull().sum())
#Substituição de valores nulos na coluna age por mediana da mesma.
dados2['Age'] = dados2['Age'].fillna(dados2['Age'].median())

#Substituição de valores nulos na coluna Fare por mediana da mesma.
dados2['Fare'] = dados2['Fare'].fillna(dados2['Fare'].median())
#matriz
previsores2 = dados2.iloc[:,[1, 3, 4, 8]].values

#Transformando a coluna sex em coluna numerica.
previsores2[:,1] = labelencoder.transform(previsores2[:,1])

#Predição do dataset2
sobreviventes2 = xg.predict(previsores2)
print(sobreviventes2)

classification = classification_report(y_teste, previsores2)
print(classification)

# Adicionar a coluna 'Survived' com a previsão
dados2['Survived'] = sobreviventes2

#Seleção de colunas
dados2 = dados2[['Survived', 'PassengerId']]

# Salvar o novo dataset com a previsão no arquivo CSV
dados2.to_csv('teste_psclass_fare_age_sex_columns.csv', index=False)

# Exibir as primeiras linhas do novo dataset com a coluna 'Survived'
print(dados2.head())