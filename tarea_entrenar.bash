#!/bin/bash






# Crear el venv de python3.10
if [ -d venv/ ]
then
  echo "Ya hay un venv creado"
else
  python3 -m venv venv
  echo "Se ha creado el venv"
fi

source venv/bin/activate

pip install -r requirements.txt

# Ingresar cuantos núcleos se van a usar
echo "Ingrese el porcentaje de nucleos que se van a usar: "
read Porcencore

# Se busca cuantos cores tiene la máquina
Ncores=$(nproc --all)

# Vamos a generar los datos que se van a procesar en
# un solo nucleo que van de 0.1% al 2.5%
Datos_Primer_core=$(python3 Primer_core.py)
$(python3 Primer_core.py > primercore.txt)


# Se va a generar los porcentajes que se van a usar en 
# el resto de núcleos
Datos_Demas_cores=$(python3 Demas_cores.py $Ncores $Porcencore)
$(python3 Demas_cores.py $Ncores $Porcencore > demascores.txt)

# Do N-1 cores at a time
# Primero se hace el entrenamiento del modelo 1
i=0
for p in $Datos_Demas_cores;
do
  python3 generar_modelo1.py $p &
  pids[${i}]=$!
  let i=$i+1
  echo i
done

# En esta seccion se hace e uso de un solo core con porcentajes de
# 0.1 a 2.5
for p in $Datos_Primer_core;
do
  python3 generar_modelo1.py $p
done 

# Esperar a todos los pids
for pid in ${pids[@]}; 
do
  wait $pid
done

# Segundo se hace el entrenamiento del modelo 2
i=0
for p in $Datos_Demas_cores;
do
  python3 generar_modelo2.py $p &
  pids[${i}]=$!
  let i=$i+1
  echo i
done

# En esta seccion se hace e uso de un solo core con porcentajes de
# 0.1 a 2.5
for p in $Datos_Primer_core;
do
  python3 generar_modelo2.py $p
done 

# Esperar a todos los pids
for pid in ${pids[@]}; 
do
  wait $pid
done

# Ahora se hace la validacion de los modelos generados 1
# Para esto vamos hacerlo en paralelo pero el método de parada 
# va a ser al esperar que todos los procesos terminen
i=0
for p in $Datos_Demas_cores;
do
  python3 Validacion.py 1 $p &
  pids[${i}]=$!
  let i=$i+1
done

# En esta seccion se hace e uso de un solo core con porcentajes de
# 0.1 a 2.5
for p in $Datos_Primer_core;
do
  python3 Validacion.py 1 $p
done

# Esperar a todos los pids
for pid in ${pids[@]}; 
do
  wait $pid
done

# Volvemos hacer lo anterior pero ahora para los modelos 2
i=0
for p in $Datos_Demas_cores;
do
  python3 Validacion.py 2 $p &
  pids[${i}]=$!
  let i=$i+1
done

# En esta seccion se hace e uso de un solo core con porcentajes de
# 0.1 a 2.5
for p in $Datos_Primer_core;
do
  python3 Validacion.py 2 $p
done

# Esperar a todos los pids
for pid in ${pids[@]}; 
do
  wait $pid
done

echo "¡¡¡¡¡¡¡¡¡¡¡¡ He terminado de hacer todo lo que me ordenaron !!!!!!!!!!!!"
