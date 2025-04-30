from src.models.model_xgboost import *
import numpy as np

#Previsao

# Obter as probabilidades preditas para todas as classes
pred_proba_xg = modelo.predict_proba(x_test)

# Previsões padrão com o maior valor de probabilidade (sem ajuste de threshold)
pred_labels_xg = np.argmax(pred_proba_xg, axis=1)