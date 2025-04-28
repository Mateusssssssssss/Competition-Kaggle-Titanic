from notebooks.eda_train import *
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
# formato da matriz
previsores = dados.iloc[:,[2, 4, 5, 6, 7, 9]].values
classe = dados.iloc[:, 1].values

labelencoder = LabelEncoder()
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])


x_train, x_test, y_train, y_test = train_test_split(previsores,
                                                        classe,
                                                        test_size = 0.4,
                                                        random_state = 1  
                                                        )