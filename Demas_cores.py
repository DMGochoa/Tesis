import sys
import numpy as np

# Ingresa la cantidad de cores que tiene el sistema y se resta e lque se usa
# entre 0.1 a 2.5
cores = (int(sys.argv[1]) - 1)

# Ingresa el procentaje de cores que se piensa usar
porcen_cores = float(sys.argv[2])

# Se identifica si ingresan el porcentaje o la proporciĂłn
if porcen_cores > 1:
    porcen_cores = porcen_cores / 100

# Se calcula el nĂşmero de cores que se van a usar
cores_para_usar = int(porcen_cores * cores)

inicio = 5
fin = 35
h = (fin - inicio) / cores_para_usar

# Ahora se tiene los porcentajes que se van a usar para el resto de cores
porcentajes_entrenar = np.arange(inicio, fin, h)

# String en donde se agrega la informaciĂłn para entregar al bash
salida = str()

for i, j in zip(range(porcentajes_entrenar.__len__()), porcentajes_entrenar):
    if i == 0:
        salida = str(j)[0:3]
    else:
        salida = salida + ' ' + str(j)[0:6]

print(salida)
