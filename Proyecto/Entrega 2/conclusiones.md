# Conclusiones

> ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de *tracking* y despliegue?

Agregar herramientas de tracking y despligue permitieron que el proyecto automatice sus procesos y se pueda actualizar de manera periódica a medida que se agregan nuevos datos de transacciones. Por otro lado, se puede realizar un seguimiento continuo de los pasos del proceso, permitiendo manejar dependencias de forma intuitiva y pudiendo corregir cualquier error sin alterar el resto de la cadena productiva.

> ¿Qué aspectos del despliegue con `Gradio/FastAPI` fueron más desafiantes o interesantes?

Conseguir que la aplicación funcione de manera adecuada y establecer una compatibilidad entre los datos ingresados por el usuario y el correcto funcionamiento de la información a desplegar.

> ¿Cómo aporta `Airflow` a la robustez y escalabilidad del pipeline?

Airflow aporta robustez al pipeline productivo al permitir definir flujos de trabajo por medio de DAGs con altos grados de modularización, lo que facilita la identificación de errores y permite reiniciar tareas específicas y manejar dependencias de forma clara, sin detener todo el proceso productivo.

En cuanto a escalabilidad, la posibilidad de ejecutar tareas en paralelo permite que se pueda trabajar con grandes volúmenes de datos, lo que es de vital importancia en un proyecto como este, donde hay tantas combinaciones cliente-producto-semana-año a ser consideradas.

> ¿Qué se podría mejorar en una versión futura del flujo? ¿Qué partes automatizarían más, qué monitorearían o qué métricas agregarían?

Se podría mejorar el proceso de extracción de datos, de forma que se recuperen de una nube en vez de trabajar con los datasets en el directorio de trabajo. Se podrían agregar además predicciones de volúmenes de compra, no sólo si un cliente va a comprar un determinado producto o no.
