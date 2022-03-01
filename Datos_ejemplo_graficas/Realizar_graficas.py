"""
En este modulo se va a realizar la creación de las gráficas resultantes de los experimentos realizados
en el entrenamiento de los mode los para la predicción de pérdidas.

Diego Alejandro Moreno Gallón
C.C. 1.088.344.588
Universidad Tecnológica de Pereira
Ingeniería Eléctrica
01/03/2022
"""

from turtle import position
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def cajas(datos=list(), xticks=list(), title='', label=tuple()):
    """
        Función para hacer la gráfica de cajas de cada porcentaje de entrenamiento.
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    y = list()
    x = list()
    for i, j in zip(datos, xticks):#range(datos.__len__())
        y.append(np.mean(i))
        x.append(float(j))
    ax.boxplot(datos, positions=x)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.plot(x, y)
    plt.title(title, fontsize=18, fontweight="bold")
    try:
        plt.xlabel(label[0])
        plt.ylabel(label[1])
    except:
        pass
    plt.show()


if __name__ == '__main__':
    # Se agregan los caminos en donde se guardan los porcentajes que se utilizaron.
    path_core1 = 'primercore.txt'
    path_cores = 'Demascores.txt'

    # Se cargan los porcentajes y luego se hace una lista con estos.
    Porcen = (open(path_core1, 'r').read() + open(path_cores, 'r').read()).replace('\n', ' ').split(' ')
    Entre_model = list()
    time_model = list()
    #print(Porcen)

    for i in Porcen:
        df = pd.read_csv('Modelos/Modelo1/scores_para_' + i + '_porciento.csv', index_col=0)
        df0 = pd.read_csv('Modelos/Modelo1/tiempos_para_' + i + '_porciento.csv', index_col=0)
        Entre_model.append((df['Validaciones']*100).to_list())
        time_model.append((df0['Tiempos']/60).to_list())
    
    #print(Entre_model)
    #print(time_model)

    cajas(Entre_model, xticks=Porcen, title="Score de entrenamiento", label=('Porcentaje Base de Datos Entrenamiento',
     'Score'))
