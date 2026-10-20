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

- `index.html` presenta los resultados guardados en `data/observatorio.json`.
- El archivo JSON **no contiene datos simulados**. Antes de ejecutar el proceso de actualización, el tablero muestra que los resultados aún no han sido publicados.
- `refresh_static_data.py` descarga y valida los archivos SIPSA-P públicos, selecciona dos series con cobertura temporal compartida, geocodifica mercados si no hay coordenadas en caché, consulta NASA POWER y estima el modelo de panel SpVARX.
- GitHub Actions ejecuta el proceso una vez al mes o cuando se inicia manualmente. Si termina correctamente, publica `data/observatorio.json` en este Space. Si falla, no sustituye los datos por datos sintéticos.


## Fuentes

- DANE, catálogo SIPSA-P 2013–2024: https://microdatos.dane.gov.co/index.php/catalog/776/get-microdata
- NASA POWER, API mensual: https://power.larc.nasa.gov/docs/services/api/temporal/monthly/
- OpenStreetMap/Nominatim para coordenadas aproximadas de mercados cuando no exista una coordenada en caché. Las coordenadas deben revisarse antes de utilizar conclusiones espaciales.

## Alcance y cautelas

- El catálogo SIPSA-P enlazado cubre 2013–2024; el pipeline predeterminado utiliza 2019–2024. Verifique la fuente para detectar futuras ediciones.
- NASA POWER proporciona datos meteorológicos/modelados de su sistema de datos, no observaciones directas de una estación en cada mercado.
- El modelo se estima como panel SpVARX con rezagos espaciales de las variables endógenas (`W y[t-1]`), no con un rezago espacial simultáneo (`W y[t]`).
- Los resultados son exploratorios/asociativos. La IRF es puntual y no tiene bandas de confianza; no debe interpretarse causalmente sin una estrategia de identificación y una inferencia más completa.
- La selección automática de productos es una configuración inicial basada en cobertura común. Para un estudio sustantivo, revise los productos, unidades, variedades, mercados y coordenadas.
