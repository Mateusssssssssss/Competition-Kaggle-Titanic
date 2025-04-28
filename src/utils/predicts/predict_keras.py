from src.models.model_keras import *
import numpy as np
# Obter as probabilidades preditas
pred_proba = classifier.predict(x_test)

# Para classificação binária:
pred_labels = (pred_proba > 0.5).astype(int).flatten()  # Se prob > 0.5, é 1. Senão, 0.


