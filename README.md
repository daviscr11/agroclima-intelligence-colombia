---
title: AgroClima Intelligence Colombia
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

# AgroClima Intelligence Colombia — Static

Panel exploratorio basado en fuentes reales: precios mayoristas SIPSA-P del DANE y variables climáticas mensuales de NASA POWER. El frontend se sirve como HTML estático; no utiliza Gradio ni necesita cómputo en el Space.

## Cómo funciona

- `index.html` presenta los resultados guardados en `data/observatorio.json`, una lectura automática de los rezagos espaciales, la auditoría mensual de Moran, los mercados y mapas departamentales de errores de ajuste y validación temporal.
- El archivo JSON **no contiene datos simulados**. Antes de ejecutar el proceso de actualización, el tablero muestra que los resultados aún no han sido publicados.
- `refresh_static_data.py` descarga y valida los archivos SIPSA-P públicos, selecciona dos series con cobertura temporal compartida, geocodifica mercados si no hay coordenadas en caché, consulta NASA POWER y estima el modelo de panel SpVARX junto con referencias espaciales SAR, SEM y SAC.
- GitHub Actions ejecuta el proceso una vez al mes o cuando se inicia manualmente. Si termina correctamente, publica en este Space la interfaz, `data/observatorio.json`, `data/residual_moran_panel_audit.csv`, `data/spvarx_rolling_backtest.csv`, los límites departamentales y el contorno nacional. Si falla, no sustituye los datos por datos sintéticos.


## Fuentes

- DANE, catálogo SIPSA-P 2013–2024: https://microdatos.dane.gov.co/index.php/catalog/776/get-microdata
- NASA POWER, API mensual: https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
- OpenStreetMap/Nominatim para coordenadas aproximadas de mercados cuando no exista una coordenada en caché. Las coordenadas deben revisarse antes de utilizar conclusiones espaciales.
- Contorno nacional: Natural Earth, Admin 0 Countries a escala 1:110m, versión 5.1.1 (dominio público): https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
- Límites departamentales: Marco Geoestadístico Nacional 2025 del DANE; versión GeoJSON simplificada distribuida por mapas-colombia (CC BY 4.0): https://geoportal.dane.gov.co/servicios/descarga-y-metadatos/datos-geoestadisticos/?cod=111 · https://github.com/ytolosa/mapas-colombia

## Alcance y cautelas

- El catálogo SIPSA-P enlazado cubre 2013–2024; el pipeline predeterminado utiliza 2019–2024. Verifique la fuente para detectar futuras ediciones.
- NASA POWER proporciona datos meteorológicos/modelados de su sistema de datos, no observaciones directas de una estación en cada mercado.
- El modelo se estima como panel SpVARX con rezagos espaciales de las variables endógenas (`W y[t-1]`), no con un rezago espacial simultáneo (`W y[t]`).
- En “Modelo espacial” se comparan complementariamente el SpVARX principal y ecuaciones de panel SAR, SEM y SAC para cada producto. Comparten observaciones, regresores locales/climáticos y efectos fijos de mercado y tiempo; SAR agrega `W y[t]`, SEM modela error espacial y SAC combina ambos. Son referencias por ecuación, no VAR multivariados alternativos.
- La tabla de comparación usa AIC gaussianos perfilados sobre la misma variable y muestra. Un AIC menor representa mejor ajuste relativo en muestra bajo los supuestos de cada especificación; no demuestra capacidad predictiva ni causalidad. `rho` y `lambda` de SAR/SEM/SAC tampoco son directamente equivalentes a los coeficientes dinámicos `W y[t-1]` del SpVARX.
- Los resultados son exploratorios/asociativos. Las respuestas impulso incluyen intervalos percentiles aproximados del 95% por horizonte mediante bootstrap circular de bloques temporales que remuestrea meses completos con todos los mercados y vuelve a estimar el SpVARX. Son intervalos punto a punto, no bandas simultáneas para toda la curva; condicionan en la matriz de pesos, productos, mercados y regresores observados, no corrigen el posible sesgo del panel dinámico ni identifican causalidad.
- El panel “Robustez espacial” reestima el mismo SpVARX con distintas cantidades de vecinos KNN (k=3, 4 y 5), manteniendo mercados, productos y meses fijos. La red se simetriza y estandariza por filas; por eso el grado final de un mercado puede exceder el k solicitado. Los intervalos de IRF son condicionales a cada matriz y la comparación no constituye una prueba formal de igualdad entre especificaciones.
- La selección automática de productos es una configuración inicial basada en cobertura común. Para un estudio sustantivo, revise los productos, unidades, variedades, mercados y coordenadas.
- El mapa de puntos muestra mercados georreferenciados y no representa una interpolación. El mapa coroplético resume por departamento los residuos del ajuste en muestra: sesgo medio con signo, MAE y RMSE. Los departamentos sin mercados observados se muestran sin datos; no se interpolan valores.
- MAE y RMSE son medidas descriptivas dentro de muestra, no errores de pronóstico fuera de muestra. El sesgo puede acercarse a cero por cancelación de errores positivos y negativos; para comparar la magnitud del error, revise MAE o RMSE. Las métricas están en unidades de Δlog(precio).
- La validación usa origen móvil de un paso y compara el SpVARX predictivo con los pronósticos de cambio cero y persistencia. Reestima usando solo los meses anteriores y no incluye efectos fijos de tiempo, que no se conocen para el mes futuro. Asume anomalía climática contemporánea cero y usa el clima rezagado observado: evalúa una proyección bajo clima normal, no un sistema de pronóstico meteorológico operativo. La pareja de productos se seleccionó por cobertura en el periodo completo y se mantiene fija en los cortes de validación.
- La comparación de MAE y RMSE agrega las pérdidas de los mercados por mes de prueba y calcula intervalos bootstrap circulares por bloques de meses, conservando juntos los mercados de cada origen. Con pocos meses de prueba los intervalos son exploratorios; si incluyen cero, no hay una diferencia clara frente a la referencia. El mapa de validación puede mostrar la mejora porcentual frente a cambio cero o persistencia, o el error absoluto departamental del SpVARX (MAE/RMSE). Los departamentos con menos de 12 predicciones se marcan sin cobertura suficiente.
- El panel “Modelo espacial” interpreta rezagos temporales y espaciales con intervalos y p-valores. Los rezagos espaciales son asociaciones condicionales de cambios de precio de mercados vecinos, no efectos causales ni impactos contemporáneos; el radio espectral informa estabilidad de la transición, no precisión predictiva.
- En “Modelo espacial” se genera una conclusión automática que sintetiza estabilidad, evidencia de rezagos espaciales, respuestas climáticas a h=1, sensibilidad de W, validación temporal y autocorrelación residual. Se actualiza con el JSON publicado; la comparación de pronósticos es frente a referencias simples y no demuestra el aporte incremental de la parte espacial.
