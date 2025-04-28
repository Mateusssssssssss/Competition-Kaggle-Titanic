from notebooks.eda_train import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Matrix
previsores = dados.iloc[:,[2, 4, 5, 6, 7, 9]].values
classe = dados.iloc[:, 1].values

labelencoder = LabelEncoder()
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])

# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    previsores, classe, test_size=0.4, random_state=1
)

#StandardScaler(): padroniza os dados, transformando-os para ter média 0 e desvio padrão 1. 
# Isso melhora o desempenho de modelos de machine learning que são sensíveis a escalas diferentes, 
# como SVM e redes neurais. 
# Padronização z-score
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)