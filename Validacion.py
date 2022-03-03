"""
Este modulo es para hacer la validación cruzada de los modelos generados para la tesis de pregrado de ingeniería eléctrica

Diego Alejandro Moreno Gallón
C.C. 1.088.344.588
Universidad Tecnológica de Pereira
Ingeniería Eléctrica
29/12/2021
"""
import sys
import time
import pickle
import numpy as np
import random as rnd
import pandas as pd
# import matplotlib.pyplot as plt


def cargar_modelo(pkl_file=str()):
    """
    Funcion para cargar el modelo
    :param pkl_file: String con ruta y nombre de archivo para cargar modelo
    :return:
    """
    with open(pkl_file, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def cargar_filas_train(fila_file=str()):
    """
    Función para cargar archivo .csv con las filas que se usaron por iteración
    :param fila_file: String con ruta del .csv con la info de las filas
    :return:
    """
    filatrain = pd.read_csv(fila_file, index_col=0)
    return filatrain


def scalamiento(base):
    """
    Funcion para el escalamiento
    :param base: Base de datos a escalar
    :return:
    """
    columnas = base.columns  # Se van a usar de la columna 3 al final

    # Se realiza un limpiado de la base de datos ya que al inicio no se limpiaron cosas que ahora se tienen
    # en cuenta para realizar este trabajo
    # Eliminar en donde no hay información de todas las columnas
    base = base.loc[~(base[columnas[3]].isnull() & base[columnas[7]].isnull() & base[columnas[10]].isnull())]

    # En los casos donde no hay voltaje pero si hay corriente se clasifica como sospechoso.
    base.loc[base[columnas[10]] == 0, 'Etiquetas'] = 1
    base.loc[base[columnas[11]] == 0, 'Etiquetas'] = 1
    base.loc[base[columnas[12]] == 0, 'Etiquetas'] = 1

    # Cualquier caso en el que hay un NaN pero si hay info en el resto es sospechoso
    base.loc[base.isnull().any(axis=1), 'Etiquetas'] = 1

    # Se eliminan las instalaciones en las que se tienen transformadores que tienen un voltaje linea-neutro
    # superior a 240 voltios ya que estos pertenencen a un grupo distinto de transformadores y la mayoría
    # de transformadores tienen salidas entre 120 y 140 voltios
    for i in (
            20417813, 20236157, 20417847, 20417843, 20236235, 19111466, 19111471, 19111422, 19111298, 20128616,
            19111467,
            20236149, 20236253, 20128604, 20730925):
        base = base.loc[~(base['Meter No.'] == i)]

    base = base.reset_index(drop=True)

    # Como se eliminaron los casos en los que no se tenia información en todas las carácteristicas ya que este caso se
    # repite bastante se agrega en caso de manera particular

    base.loc[base.shape[0]] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    base = base.fillna(0)

    # se divide por el maximo y z-score
    for i in columnas[3:10]:
        base[i] = base[i] / base[i].mean()
    for i in columnas[10:13]:
        base[i] = (base[i] - base[i].mean()) / base[i].std()

    return base


def base_test(base, indices1, indices0):
    """
    Funcion para eliminar las filas que se usaron para entrenar
    :param base: Base que se va a eliminar
    :param indices: Lista de indices con los a eliminar
    :return:
    """
    columnas = base.columns  # Se van a usar de la columna 3 al final

    # Se seleccionan los datos normales y anormales
    anormales = base.loc[base[columnas[13]] == 1]
    normales = base.loc[base[columnas[13]] == 0]

    # Se dividen los datos en caracteristicas y etiquetas
    X1 = anormales[columnas[3:13]].to_numpy()
    Y1 = anormales[columnas[13]].to_numpy()
    X0 = normales[columnas[3:13]].to_numpy()
    Y0 = normales[columnas[13]].to_numpy()

    # Se eliminan las partes que fueron usadas para el entrenamiento
    X1 = np.delete(X1, indices1, axis=0)
    Y1 = np.delete(Y1, indices1, axis=0)
    X0 = np.delete(X0, indices0, axis=0)
    Y0 = np.delete(Y0, indices0, axis=0)

    return X1, Y1, X0, Y0


def random_Vn(X, Ktest, iteraciones):
    """
    Funcion que crea todos los indices que se van a usar para el entrenamiento y validacion.
    :param X: Bases
    :param Ktrain: Porcion que se va usar para entrenar
    :param Ktest: Porcion que se va usar para validar
    :param Iteraciones: Iteraciones máximas
    :return: Indices
    """
    arr = np.arange(0, X.shape[0])
    Nmuestras = int(1 / Ktest)
    tamano = int(X.shape[0] * Ktest)
    Ntotal = 0

    while Ntotal < iteraciones:
        aleatorio = np.random.permutation(arr)
        aleatorio = aleatorio[0: tamano * Nmuestras].reshape(-1, tamano)
        if Ntotal == 0:
            salida = aleatorio
        else:
            salida = np.concatenate((salida, aleatorio), axis=0)
        Ntotal = salida.shape[0]
        if Ntotal > iteraciones:
            print(salida.shape)
            break
    return salida


def Conjunto_datos_Vn(X1, Y1, X0, Y0, Ktest, iteraciones):
    filasX1 = random_Vn(X=X1, Ktest=Ktest, iteraciones=iteraciones)
    filasX0 = random_Vn(X=X0, Ktest=Ktest, iteraciones=iteraciones)
    return X1, Y1, X0, Y0, filasX1, filasX0


def Seleccion_datos_iter(x1, y1, x0, y0, filasX1, filasX0, iter):
    filaselec1 = filasX1[iter].tolist()
    filaselec0 = filasX0[iter].tolist()

    X_test1 = x1[filaselec1]
    y_test1 = y1[filaselec1]

    X_test0 = x0[filaselec0]
    y_test0 = y0[filaselec0]

    # Se unen para crear los datos de entrenamiento y los de validación
    X_test = np.concatenate((X_test1, X_test0), axis=0)
    y_test = np.concatenate((y_test1, y_test0), axis=0)

    return X_test, y_test


if __name__ == '__main__':
    quemodelo = int(sys.argv[1])

    if quemodelo == 1:
        base = pd.read_csv('Base_etiquetada.csv', index_col=1)
        base = base.reset_index(drop=True)
        base = scalamiento(base)
    elif quemodelo == 2:
        base = pd.read_csv('Base2_etiquetada.csv')
        base = base.reset_index(drop=True)
        base = scalamiento(base)
    guardado_de_informacion = {'Tiempo': [], 'Modelo': [], 'Score': [], 'Promedio': []}
    Ktest = 0.05
    promedio_ = np.array([])
    stdesv = np.array([])
    # porcen = 10 # float(sys.argv[1])
    scores_std = np.array([])
    scores = np.array([])
    # rev es un lista que me indica los porcentajes que se van analizar
    rev = [float(sys.argv[2])]

    for porcen in rev:
        porcion = porcen / 100

        entrenamiento0 = cargar_filas_train(
            'Modelos/modelo' + str(quemodelo) + '/datos0_para_' + str(porcen) + '_porciento.csv')
        entrenamiento1 = cargar_filas_train(
            'Modelos/modelo' + str(quemodelo) + '/datos1_para_' + str(porcen) + '_porciento.csv')
        promedio_modelos = cargar_filas_train(
            'Modelos/modelo' + str(quemodelo) + '/promedios_para_' + str(porcen) + '_porciento.csv')
        scores_modelos = cargar_filas_train(
            'Modelos/modelo' + str(quemodelo) + '/scores_para_' + str(porcen) + '_porciento.csv')
        tiempos_modelos = cargar_filas_train(
            'Modelos/modelo' + str(quemodelo) + '/tiempos_para_' + str(porcen) + '_porciento.csv')

        canti_modelos = promedio_modelos.shape[0]
        iteraciones = 30  #

        for i in range(canti_modelos):
            instanteInicio = time.time()
            modelo = cargar_modelo(pkl_file='Modelos/modelo' + str(quemodelo) + '/modelo' + str(i) + '_para_' + str(
                porcen) + '_porciento.pkl')
            X1, Y1, X0, Y0 = base_test(base, entrenamiento1['iter' + str(i)].to_list(),
                                       entrenamiento0['iter' + str(i)].to_list())
            filas1 = random_Vn(X=X1, Ktest=Ktest, iteraciones=iteraciones)
            filas0 = random_Vn(X=X0, Ktest=Ktest, iteraciones=iteraciones)

            validaciones = np.array([])
            for iter in range(iteraciones):
                X_test, y_test = Seleccion_datos_iter(x1=X1, y1=Y1, x0=X0, y0=Y0,
                                                      filasX1=filas1, filasX0=filas0,
                                                      iter=iter)
                valor = modelo.score(X_test, y_test)
                validaciones = np.append(validaciones, valor)
                print('[', str(porcion * 100), '%] t=', "%.2f" % ((time.time() - instanteInicio) / 60.0),
                      ' minutos. iter=', iter, '. modelo=', i, '. Score=', "%.2f %%" % (valor * 100.0),
                      flush=True)

                # Se procede a guardar la información que luego se va usar para luego graficar
                guardado_de_informacion['Tiempo'].append((time.time() - instanteInicio) / 60.0)
                guardado_de_informacion['Modelo'].append(i)
                guardado_de_informacion['Score'].append(valor)
                guardado_de_informacion['Promedio'].append(validaciones.mean())

            scores = np.append(scores, validaciones.mean() * 100)
            scores_std = np.append(scores_std, validaciones.std() * 100)

        promedio_ = np.append(promedio_, scores.mean())
        stdesv = np.append(stdesv, scores.std())

    res = pd.DataFrame(
        {'Porcentaje_Train': rev, 'Promedio': promedio_.tolist(), 'StandarDesv': stdesv.tolist()})
    res.to_csv('Resultados/modelo' + str(quemodelo) + '/resultados_validaciones' + str(porcion * 100) + '.csv')

    datos_val = pd.DataFrame(guardado_de_informacion)
    datos_val.to_csv(
        'Resultados/modelo' + str(quemodelo) + '/Datos_validacion_' + str(porcion * 100) + '_para' + str(
            canti_modelos) + '_modelos' + '.csv')
    print('[', str(porcen), '%] - FIN.', flush=True)

    # plt.figure(0)
    # plt.errorbar(res['Porcentaje_Train'], res['Promedio'], res['StandarDesv'])
    # plt.errorbar(np.arange(canti_modelos), scores, scores_std)
    # plt.grid()
    # plt.xlabel('Porcentaje')
    # plt.ylabel('Score')
    # plt.savefig('Entrenamiento.svg')
    # plt.show()
