from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from notebooks.preprocess import * 

modelo = XGBClassifier(objective='binary:logistic',  # Classificação binária
    eval_metric='auc',            # Métrica de avaliação
    n_estimators=1500,             # Número de árvores
    learning_rate=0.01,           # Taxa de aprendizado
    max_depth=45,                  # Profundidade das árvores
    subsample=0.5,                # Amostragem para evitar overfitting
    colsample_bytree=0.8,         # Porcentagem de colunas usadas
    gamma=1,                      # Evita overfitting
    reg_lambda=1,                 # Regularização L2
    reg_alpha=1,                   # Regularização L1
    
)

#Validação cruzada
results = cross_val_score(modelo, x_train, y_train, cv=10)
print(f'Cross Validation: {results}')

modelo.fit(x_train, y_train)