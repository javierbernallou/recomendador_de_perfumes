# Recomendador de Perfumes

## Descripción general
Proyecto de recomendación de perfumes basado en análisis de similitud entre fragancias. El objetivo es identificar perfumes con perfiles olfativos parecidos y sugerir opciones adecuadas según características sensoriales y atributos del producto.

## Objetivo
Ayudar a usuarios a descubrir fragancias afines a su gusto o a un perfume de referencia mediante técnicas de recomendación y análisis de datos del mercado de perfumes.

## Stack principal
- Lenguaje: Python
- Entorno: Jupyter Notebooks
- Bibliotecas principales: pandas, NumPy, scikit-learn y librerías para análisis de texto si se procesan descripciones olfativas

## Estructura del proyecto
```bash
notebooks/           Análisis y experimentación del modelo
data/                Datasets de perfumes y atributos
authors/             Recursos auxiliares o notebooks complementarios
src/                 Funciones utilitarias y preprocesado
requirements.txt     Dependencias del proyecto
README.md            Documentación del repositorio
```

## Cómo ejecutar
1. Clonar el repositorio.
2. Crear entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Abrir el notebook principal y ejecutar:
   - carga del dataset,
   - limpieza y preparación de atributos,
   - cálculo de similaridad o recomendación,
   - validación del resultado.

## Casos de uso
- recomendación basada en perfiles olfativos,
- comparación entre fragancias,
- descubrimiento de aromas similares,
- apoyo a plataformas de e-commerce o recomendación de productos.

## Recomendaciones
Para una versión de producción, el modelo puede evolucionar hacia una API de recomendación o un sistema híbrido que combine preferencias del usuario con atributos de los perfumes.
