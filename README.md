# Deep learning sobre imágenes satelitales: evaluación de daños por desastres naturales

Dos proyectos de la asignatura *Aprendizaje Profundo para el Análisis de Imágenes* (Grado en Ingeniería de Tecnologías de Telecomunicación, UC3M, curso 2025/26), sobre un subconjunto del dataset **xBD** de imágenes satelitales antes y después de desastres.

## 1. Clasificación del daño en edificios — [`clasificacion/`](clasificacion/)
CNN para clasificar el nivel de daño de cada edificio a partir de recortes de imágenes satelitales.
- Carga de una base de datos propia en PyTorch (`Dataset`/`DataLoader`), recorte de edificios a partir de polígonos WKT, aumento de datos.
- Red propia y *fine-tuning* de modelos preentrenados de `torchvision`, entrenamiento con precisión mixta y *scheduler* de tasa de aprendizaje.
- Evaluación con matriz de confusión y F1 por clase sobre un conjunto desbalanceado (ver `class_distribution.png`).

## 2. Segmentación semántica con DeepLab-v3 — [`segmentacion/`](segmentacion/)
*Fine-tuning* de `deeplabv3_resnet101` para segmentar edificios y su nivel de daño píxel a píxel.
- Adaptación de la cabeza de la red (`DeepLabHead`, `FCNHead`) al número de clases.
- Pérdidas para clases desbalanceadas (incluida *focal loss*) y métricas IoU/Jaccard y AUC.
- Informe con resultados y análisis en [`segmentacion/informe.pdf`](segmentacion/informe.pdf).

## Ejecutar
Los cuadernos se ejecutaron en Google Colab con GPU. Necesitan PyTorch, torchvision, scikit-learn, scikit-image, shapely, OpenCV y natsort. El dataset y los pesos entrenados no se incluyen por tamaño.

*Los enunciados y la estructura de los cuadernos son del profesorado de la asignatura; la implementación y los resultados son míos.*
