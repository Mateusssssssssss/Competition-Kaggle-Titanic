from data.dados import load_data
import matplotlib.pyplot as plt
import seaborn as sb

dados = load_data()

print(dados.head())
print(dados.describe())
print(dados.shape)
# Quantidade de nulos
# substituindo os nulos da coluna Age pela mediana
dados['Age'] = dados['Age'].fillna(dados['Age'].median())

#Verificação de outliers
nulls = dados.isnull().sum()
print(f'Quantidade de Nulos:\n{nulls}')

sb.boxplot(dados[['Pclass', 'Fare']])
plt.show()
#Verificando moda da coluna Age
age_moda = dados['Age'].mode()
print(f'Moda da idade{age_moda}')

qt_class = dados['Survived'].value_counts()
print(f'Quantidade de Sobreviviente e não Sobreviventes\n{qt_class}')

qtd_class = dados['Survived'].nunique()

#Analise de correlação
correlacao = dados[['Survived', 'Age', 'Parch', 'Pclass', 'Parch','Fare','SibSp']].corr()
print(correlacao)

