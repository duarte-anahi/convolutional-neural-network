# Modelo convolucional: diseño, entrenamiento y diagnóstico

[English](README.md) · **Español**

Una red neuronal convolucional construida con Keras sobre un conjunto de imágenes de 8 clases. Lo que importa del proyecto no es la precisión sino el **diagnóstico**: por qué el modelo no generaliza y por qué achicarlo no lo resolvió.

![Curvas de entrenamiento de las dos versiones](docs/comparison.png)

## Arquitectura

| Bloque | Filtros | Salida |
|---|---|---|
| 1 | 32 | 100×100×32 |
| 2 | 64 | 50×50×64 |
| 3 | 128 | 25×25×128 |
| 4 | 128 | 12×12×128 |

Cuatro bloques `Conv2D` + `MaxPooling2D`, después `Flatten`, una capa densa de 128 unidades y una softmax de 8 salidas: **~2,6M de parámetros**. Entrenado con Adam y `sparse_categorical_crossentropy` durante 15 épocas.

**Datos:** 1.567 imágenes redimensionadas a 200×200, divididas 80/20 (1.254 de entrenamiento, 313 de validación).

## El experimento

Una segunda versión mantiene **32 filtros en todos los bloques** —~619K parámetros, 75% menos— para poner a prueba si el sobreajuste venía del exceso de capacidad.

| Modelo | Parámetros | Accuracy entrenamiento | Accuracy validación | Loss validación |
|---|---|---|---|---|
| V1 · filtros crecientes | ~2,6M | 0,992 | 0,597 | 2,63 |
| V2 · filtros constantes | ~619K | 0,994 | 0,585 | 2,57 |

## Diagnóstico

Las dos versiones memorizan el conjunto de entrenamiento (~99%) mientras la precisión de validación se estanca cerca del 59%, y la pérdida de validación empieza a subir en la época 6 mientras la de entrenamiento sigue bajando: la forma clásica del sobreajuste.

Recortar el 75% de los parámetros no cambió nada, así que el cuello de botella está en el **conjunto de datos** (tamaño y variedad) y no en la arquitectura.

**Lo que me dejó este proyecto:** cuando la pérdida de validación sube mientras la de entrenamiento baja, la respuesta no es otra capa sino más y mejores datos.

**Próximos pasos:** aumentación de datos y transfer learning desde un modelo preentrenado.

## Cómo correrlo

El notebook está pensado para **Google Colab**. El conjunto de datos está en Google Drive; la sección 0 del notebook explica cómo agregarlo al propio Drive antes de ejecutarlo.

## Tecnologías

Python · TensorFlow · Keras · NumPy · Matplotlib
