from src.models.model_forest import *
import numpy as np

#Previsao

# Obter as probabilidades preditas para todas as classes
pred_proba_forest = forest.predict_proba(x_test)

# Previsões padrão com o maior valor de probabilidade (sem ajuste de threshold)
pred_labels_forest = np.argmax(pred_proba_forest, axis=1)