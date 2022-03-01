"""
En este modulo se va a realizar la creación de las gráficas resultantes de los experimentos realizados
en el entrenamiento de los mode los para la predicción de pérdidas.

Diego Alejandro Moreno Gallón
C.C. 1.088.344.588
Universidad Tecnológica de Pereira
Ingeniería Eléctrica
01/03/2022
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#def 

if __name__ == '__main__':
    # Se agregan los caminos en donde se guardan los porcentajes que se utilizaron.
    path_core1 = 'primercore.txt'
    path_cores = 'Demascores.txt'

    # Se cargan los porcentajes y luego se hace una lista con estos.
    Porcen = (open(path_core1, 'r').read() + open(path_cores, 'r').read()).replace('\n', ' ').split(' ')
    print(Porcen)
    Entre_model = dict()

    for i in Porcen:
        df = pd.read_csv('scores_para_' + i + '_porciento.csv', index_col=0)
