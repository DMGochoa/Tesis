import sys

listado=str()
valor=0.1
delta=0.1#0.9
parada=0.5

# Se crea un string con los datos desde 0.1 a 3.0
while True:
    if valor == 0.1:
        listado = str(valor)[0:3]
    else:
        listado = listado + ' ' + str(valor)[0:3]

    valor=valor+delta

    if valor>=parada:
        break

print(listado)
