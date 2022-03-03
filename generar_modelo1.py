"""
Este modulo es para hacer los modelos para la tesis de pregrado de ingeniería eléctrica

Diego Alejandro Moreno Gallón
C.C. 1.088.344.588
Universidad Tecnológica de Pereira
Ingeniería Eléctrica
12/12/2021
"""
import sys
import time
import pickle
import numpy as np
import random as rnd
import pandas as pd
from sklearn.svm import SVC, LinearSVC


def entrenar_modelo(X, y, X_test, y_test, gam=0.1):
    """
    Funcion para el entrenamiento del modelo.
    :param X: Datos mxn (m muestras por n caracteristicas)
    :param y: Etiquetas mx1 (m muestras por 1)
    :param X_test: Datos de evaluación
    :param y_test: Etiquetas de los datos de evaluación
    :return:
    """
    ini = time.time()
    modelo = SVC(kernel='rbf', gamma=gam, verbose=0).fit(X, y)
    #modelo = LinearSVC(max_iter=5000, verbose=1).fit(X, y)
    # print(time.time() - ini)
    resultado = modelo.score(X_test, y_test)
    # print(time.time() - ini)

    return modelo, resultado


def random_select(X, y, Ktrain, Ktest):
    """
    Función que toma los datos de manera aleatoria de la bolsa de datos esto para probar.
    :param X: Datos de entrenamiento
    :param y: Etiquetas.
    :param Ktrain: Porcion que se va a sacar para entrenar.
    :param Ktest: Porcion que se va a usar para validar.
    :return:
    """
    # Se generan la muestra aleatoria sin repetir datos
    seleccion = rnd.sample(range(X.shape[0]), int(X.shape[0] * (Ktrain + Ktest)))
    # Cuantos y cuales son para entrenamiento y se guardan
    cantientre = int(seleccion.__len__() * Ktrain / (Ktrain + Ktest))
    fentrenar = seleccion[0: cantientre]
    # Cuales son de test
    ftest = seleccion[cantientre: seleccion.__len__()]
    # Selección de datos
    X_train = X[fentrenar]
    X_test = X[ftest]
    y_train = y[fentrenar]
    y_test = y[ftest]
    return X_train, X_test, y_train, y_test, fentrenar


def random_Vn(X, Ktrain, Ktest, iteraciones):
    """
    Funcion que crea todos los indices que se van a usar para el entrenamiento y validacion.
    :param X: Bases
    :param Ktrain: Porcion que se va usar para entrenar
    :param Ktest: Porcion que se va usar para validar
    :param Iteraciones: Iteraciones máximas
    :return: Indices
    """
    arr = np.arange(0, X.shape[0])
    Nmuestras = int(1 / (Ktrain + Ktest))
    tamano = int(X.shape[0] * (Ktrain + Ktest))
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


def scalamiento(base):
    columnas = base.columns  # Se van a usar de la columna 3 al final

    # Eliminar en donde no hay información de todas las columnas
    base = base.loc[~(base[columnas[3]].isnull() & base[columnas[7]].isnull() & base[columnas[10]].isnull())]
    base = base.drop_duplicates()
    print(base.shape[0])

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
    for i in ['Active energy(+) total(kWh)', 'Reactive energy(+) total(kvarh)',
       'Active energy(-) total(kWh)', 'Reactive energy (-) total(kvarh)',
       'current in phase L1(A)', 'current in phase L2(A)',
       'current in phase L3(A)']:
        base[i] = base[i] / base[i].max()  # .mean()
    for i in ['voltage in phase L1(V)', 'voltage in phase L3(V)',
       'voltage in phase L2(V)']:
        base[i] = (base[i] - base[i].mean()) / base[i].std()

    return base


def Conjunto_datos_Vn(base, Ktrain, Ktest, iteraciones):
    columnas = base.columns  # Se van a usar de la columna 3 al final

    # Se seleccionan los datos normales y anormales
    anormales = base.loc[base[columnas[13]] == 1]
    normales = base.loc[base[columnas[13]] == 0]

    # Se dividen los datos en caracteristicas y etiquetas
    X1 = anormales[columnas[3:13]].to_numpy()
    Y1 = anormales[columnas[13]].to_numpy()
    X0 = normales[columnas[3:13]].to_numpy()
    Y0 = normales[columnas[13]].to_numpy()

    filasX1 = random_Vn(X=X1, Ktrain=Ktrain, Ktest=Ktest, iteraciones=iteraciones)
    filasX0 = random_Vn(X=X0, Ktrain=Ktrain, Ktest=Ktest, iteraciones=iteraciones)
    return X1, Y1, X0, Y0, filasX1, filasX0


def Seleccion_datos(base, Ktrain, Ktest):
    """
    Función para partir los datos de entrenamiento y de testeo.
    :param base: Dataframe con la información.
    :param Ktrain: Porcentaje de datos para entrenamiento.
    :param Ktest: Porcentaje de datos para la validación.
    :return: Retorna X_train, y_train, X_test, y_test
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

    # Se separa equivalentemente para mantener la proporción
    X_train1, X_test1, y_train1, y_test1, filastrain1 = random_select(X=X1, y=Y1, Ktrain=Ktrain, Ktest=Ktest)
    X_train0, X_test0, y_train0, y_test0, filastrain0 = random_select(X=X0, y=Y0, Ktrain=Ktrain, Ktest=Ktest)

    # Se unen para crear los datos de entrenamiento y los de validación
    X_train = np.concatenate((X_train1, X_train0), axis=0)
    X_test = np.concatenate((X_test1, X_test0), axis=0)
    y_train = np.concatenate((y_train1, y_train0), axis=0)
    y_test = np.concatenate((y_test1, y_test0), axis=0)

    # Filas que se usaron para entrenar
    filastrain = filastrain1 + filastrain0
    return X_train, X_test, y_train, y_test, filastrain


def Seleccion_datos_iter(x1, y1, x0, y0, filasX1, filasX0, iter, Ktrain, Ktest):
    filaselec1 = filasX1[iter].tolist()
    filaselec0 = filasX0[iter].tolist()

    Porcion_train1 = int(filaselec1.__len__() * Ktrain / (Ktrain + Ktest))
    Porcion_train0 = int(filaselec0.__len__() * Ktrain / (Ktrain + Ktest))

    Datos_train1 = filaselec1[0:Porcion_train1]
    Datos_train0 = filaselec0[0:Porcion_train0]

    X_train1 = x1[Datos_train1]
    X_test1 = x1[filaselec1[Porcion_train1: filaselec1.__len__()]]
    y_train1 = y1[Datos_train1]
    y_test1 = y1[filaselec1[Porcion_train1: filaselec1.__len__()]]

    X_train0 = x0[Datos_train0]
    X_test0 = x0[filaselec0[Porcion_train0: filaselec0.__len__()]]
    y_train0 = y0[Datos_train0]
    y_test0 = y0[filaselec0[Porcion_train0: filaselec0.__len__()]]

    # Se unen para crear los datos de entrenamiento y los de validación
    X_train = np.concatenate((X_train1, X_train0), axis=0)
    X_test = np.concatenate((X_test1, X_test0), axis=0)
    y_train = np.concatenate((y_train1, y_train0), axis=0)
    y_test = np.concatenate((y_test1, y_test0), axis=0)

    return X_train, X_test, y_train, y_test, Datos_train1, Datos_train0


def main(Ktrain=0.0005, Ktest=0.005, iteraciones=150, base=pd.DataFrame()):
    """
    Función principal para realizar montecarlo
    :return:
    """

    tol = 1
    pos = 0
    iter = 0

    Xyrandom = list()
    entrenamiento1 = dict()
    entrenamiento0 = dict()
    promedio = np.array([])
    tiempos = np.array([])
    scores = np.array([])
    modelos = list()
    base = scalamiento(base)
    instante0 = time.time()
    # TODO: Esta parte es la vieja seleccion de manera aleatoria

    # for i in range(iteraciones):
    #    Xtest, Xtrain, ytest, ytrain, filastrain = Seleccion_datos(base=base, Ktrain=Ktrain, Ktest=Ktest)
    #    Xyrandom.append([Xtest, Xtrain, ytest, ytrain])

    # TODO: Esta parte es la nueva seleccion de manera aleatoria

    X1, Y1, X0, Y0, filasX1, filasX0 = Conjunto_datos_Vn(base, Ktrain, Ktest, iteraciones)

    print('[', str(Ktrain * 100), '%] t=' ,  "%.2f" % (time.time() - instante0), 'seg para crear las muestras de entrenamiento y',
            ' validación', flush=True)
    while True:

        X_train, X_test, y_train, y_test, Datos_train1, Datos_train0 = Seleccion_datos_iter(x1=X1, y1=Y1, x0=X0, y0=Y0,
                                                                                            filasX1=filasX1,
                                                                                            filasX0=filasX0, iter=iter,
                                                                                            Ktrain=Ktrain, Ktest=Ktest)
        entrenamiento0['iter' + str(iter)] = Datos_train0
        entrenamiento1['iter' + str(iter)] = Datos_train1
        # X_test = Xyrandom[iter][0]
        # X_train = Xyrandom[iter][1]
        # y_test = Xyrandom[iter][2]
        # y_train = Xyrandom[iter][3]
        inicio = time.time()

        modelo, resultado = entrenar_modelo(X=X_train, y=y_train, X_test=X_test, y_test=y_test, gam=0.1)
        fin = time.time()

        modelos.append(modelo)
        scores = np.append(scores, resultado)
        tiempos = np.append(tiempos, fin - inicio)
        prom = scores.mean()
        promedio = np.append(promedio, prom)

        print('[', str(Ktrain * 100), '%] t=', "%.2f" % ((time.time() - instante0) / 60.0),
              ' minutos. iter=', iter, '. Score=', "%.2f %%" % (prom * 100.0), flush=True)

        if pos != 0:
            tol = np.abs(prom - promedio[pos - 1])
        if ((tol < 1e-5) & (iter >= 5)) | (iter == iteraciones):  # (iter == iteraciones):
            break

        iter += 1
        pos += 1

    return modelos, scores, iter, tol, promedio, prom, [X_test, X_train, y_test, y_train, tiempos, entrenamiento0,
                                                        entrenamiento1]


if __name__ == '__main__':
    # Se trae el dato del porcentaje que va ser de validación de la terminal
    porcen = sys.argv[1] # 0.1
    porcion = float(porcen) / 100
    print('\n[', str(porcen), '%] - INICIO.', flush=True)

    # Se carga la base de datos
    base = pd.read_csv('Base_etiquetada.csv', index_col=1)
    base = base.reset_index(drop=True)

    # Número máximo de iteraciones y porcentaje de test
    iteraciones = 300
    Ktest = 5 / 100

    modelos, scores, iter, tol, promedio, prom, datos = main(Ktrain=porcion, Ktest=Ktest, iteraciones=iteraciones,
                                                             base=base)

    for i in range(datos[4].__len__()):
        pickle.dump(modelos[i],
                    open('Modelos/modelo1/modelo' + str(i) + '_para_' + str(porcen) + '_porciento.pkl', 'wb'))

    promedios = pd.DataFrame({'Promedio': promedio})
    validaciones = pd.DataFrame({'Validaciones': scores})
    tiempos = pd.DataFrame({'Tiempos': datos[4]})
    entrenamiento0 = pd.DataFrame(datos[5])
    entrenamiento1 = pd.DataFrame(datos[6])

    promedios.to_csv('Modelos/modelo1/promedios_para_' + str(porcen) + '_porciento.csv')
    validaciones.to_csv('Modelos/modelo1/scores_para_' + str(porcen) + '_porciento.csv')
    entrenamiento0.to_csv('Modelos/modelo1/datos0_para_' + str(porcen) + '_porciento.csv')
    entrenamiento1.to_csv('Modelos/modelo1/datos1_para_' + str(porcen) + '_porciento.csv')
    tiempos.to_csv('Modelos/modelo1/tiempos_para_' + str(porcen) + '_porciento.csv')

    print('[', str(porcen), '%] - FIN.', flush=True)
