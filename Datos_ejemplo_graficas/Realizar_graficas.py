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

sns.set()

def cajas(datos=list(), xticks=list(), title='', label=tuple(), figax=tuple()):
    """
        Función para hacer la gráfica de cajas de cada porcentaje de entrenamiento.
    """
    figax[0]
    figax[1]
    #fig = plt.figure(figsize=(10, 5))
    #ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

    y = list()
    x = list()
    for i, j in zip(datos, xticks):#range(datos.__len__())
        y.append(np.mean(i))
        x.append(float(j))
    resultados = figax[1].boxplot(datos, positions=x)
    figax[1].set_xticks(x)
    figax[1].set_xticklabels(x)
    figax[1].plot(x, y)
    figax[1].set_title(title, fontsize=18, fontweight="bold")
    try:
        figax[1].set_xlabel(label[0])
        figax[1].set_ylabel(label[1])
    except:
        pass

if __name__ == '__main__':
    # Se agregan los caminos en donde se guardan los porcentajes que se utilizaron.
    path_core1 = 'primercore.txt'
    path_cores = 'Demascores.txt'

    # Se cargan los porcentajes y luego se hace una lista con estos.
    Porcen = (open(path_core1, 'r').read() + open(path_cores, 'r').read()).replace('\n', ' ').split(' ')
    Entre_model = list()
    time_model = list()
    #print(Porcen)
    Score_val = list()
    Time_val = list()

    for i in Porcen:
        df = pd.read_csv('Modelos/modelo1/scores_para_' + i + '_porciento.csv', index_col=0)
        df0 = pd.read_csv('Modelos/modelo1/tiempos_para_' + i + '_porciento.csv', index_col=0)
        df1 = pd.read_csv('Resultados/modelo1/Datos_validacion_' + i + '_para' + str(df.shape[0]) + '_modelos' 
        + '.csv', index_col=0)
        Entre_model.append((df['Validaciones']*100).to_list())
        time_model.append((df0['Tiempos']/60).to_list())
        Score_val.append((df1['Score']*100).to_list())
        Time_val.append((df1['Tiempo']/60).to_list())
    
    #print(Entre_model)
    #print(time_model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 16))

    cajas(Score_val, xticks=Porcen, title="Score de Validación", 
        label=('Porcentaje Base de Datos Entrenamiento',
        'Score'), figax=(fig, axes[1]))

    cajas(Entre_model, xticks=Porcen, title="Score de Entrenamiento", 
        label=('Porcentaje Base de Datos Entrenamiento',
        'Score'), figax=(fig, axes[0]))
    axes[0].set_ylim([78, 80])
    axes[1].set_ylim([65, 80])
    

    fig0, axes0 = plt.subplots(1, 2, figsize=(14, 16))
    
    cajas(time_model, xticks=Porcen, title="Tiempo de Entrenamiento", 
        label=('Porcentaje Base de Datos Entrenamiento',
        'Tiempo [min]'), figax=(fig0, axes0[0]))

    cajas(Time_val, xticks=Porcen, title="Tiempo de Validación", 
        label=('Porcentaje Base de Datos Entrenamiento',
        'Tiempo [min]'), figax=(fig0, axes0[1]))

    plt.show()
