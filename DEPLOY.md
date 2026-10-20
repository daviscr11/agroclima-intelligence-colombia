# Publicación gratuita: Hugging Face Static + GitHub Actions

## Por qué hay dos repositorios

Un Space con `sdk: static` solo sirve archivos estáticos. No ejecuta `app.py`, Gradio, Python ni el modelo econométrico. Para mantener la interfaz gratis sin perder el cálculo SpVARX, el trabajo pesado se hace en GitHub Actions y se publica solamente el `data/observatorio.json` final en Hugging Face.

## Publicación inicial

1. Descarga y descomprime `AgroClima_Intelligence_Colombia_Static_Space.zip`.
2. Crea un repositorio en GitHub y sube TODO el contenido de esta carpeta a la raíz, incluida la carpeta oculta `.github/workflows/`.
3. Crea un Space en https://huggingface.co/new-space y elige SDK **Static**.
4. Sube `README.md`, `index.html` y la carpeta `data/` al Space. Puedes subir también los módulos y manuales; el navegador no los ejecutará, pero documentan el pipeline.
5. En GitHub, entra a Settings → Secrets and variables → Actions → New repository secret:
   - `HF_TOKEN`: token Hugging Face con permiso de escritura al Space.
   - `HF_SPACE_ID`: `usuario/nombre-del-space`.
6. En GitHub, abre Actions, habilita los workflows si te lo solicita, selecciona **Actualizar datos oficiales y publicar en Hugging Face** y pulsa **Run workflow**.
7. Revisa los logs. El proceso puede fallar de forma explícita si cambia el catálogo del DANE, si el parser no reconoce las tablas, si no hay productos con cobertura conjunta o si hay mercados sin geocodificar. No genera datos sintéticos.

## Actualización

El workflow tiene una ejecución programada mensual y también se puede ejecutar manualmente desde Actions. La frecuencia de los eventos programados de GitHub Actions no constituye una garantía de publicación en un minuto exacto. Si el paso falla, no se publica JSON nuevo.

## Configuración del modelo

Variables de entorno del workflow:

- `AGROCLIMA_FORCE_REFRESH`: por defecto el workflow fuerza la descarga de los ZIP seleccionados y su reextracción para no usar una versión vieja en caché.
- `AGROCLIMA_START_YEAR`: inicio de la ventana, valor inicial 2019.
- `AGROCLIMA_END_YEAR`: final de la ventana, valor inicial 2024, último año del catálogo SIPSA-P consultado.
- `AGROCLIMA_MIN_MARKETS`: mínimo de mercados completos, valor inicial 6.
- `AGROCLIMA_MIN_MONTHS`: mínimo de meses completos, valor inicial 36.
- `AGROCLIMA_KNN_K`: vecinos para W, valor inicial 4.
- `AGROCLIMA_IRF_HORIZON`: meses de IRF, valor inicial 18.

El pipeline autoelige dos productos diferentes con mejor cobertura común, para generar un primer panel vectorial sin inventar observaciones. Esa regla es técnica, no una recomendación económica sobre cuáles productos estudiar. Una vez identificado el fenómeno agropecuario de interés, conviene fijar explícitamente productos, unidades y variedades.

## Limitaciones metodológicas importantes

- La matriz KNN se construye con coordenadas geográficas de mercados. Las nuevas coordenadas geocodificadas por Nominatim se guardan en `data/market_coordinates.csv`; revise esa tabla y ajuste puntos incorrectos.
- Los precios SIPSA-P son precios nominales mayoristas reportados por mercado. NASA POWER proporciona variables meteorológicas derivadas de sus conjuntos de datos; interprete la resolución geográfica y temporal de forma adecuada.
- La versión actual estima las ecuaciones de un panel SpVARX(1) con rezagos espaciales de variables endógenas (`W y[t-1]`), efectos de mercado/tiempo y covarianza Driscoll–Kraay. Los sesgos posibles de un panel dinámico con efectos fijos y la inferencia de sistema requieren robustez adicional.
- La respuesta a impulsos es puntual, carece de bandas de confianza y no identifica por sí sola un efecto causal.
- El catálogo enlazado del DANE contiene SIPSA-P 2013–2024 en la revisión consultada. Cuando se publiquen nuevos años, habrá que ampliar el límite del downloader (`dane_auto.py`) y actualizar la configuración.

## Archivos

- `index.html`: aplicación estática de visualización.
- `data/observatorio.json`: resultados que lee el navegador; empieza sin datos y se regenera desde fuentes oficiales.
- `refresh_static_data.py`: descarga, selección de productos, consulta climática, estimación y creación del JSON.
- `publish_to_hf.py`: publica el artefacto JSON al Space mediante `HF_TOKEN`.
- `.github/workflows/update-data.yml`: ejecución mensual/manual en GitHub Actions.
- `dane_auto.py`, `agroclima_spvarx_real.py`: funciones de descarga, lectura y econometría.
