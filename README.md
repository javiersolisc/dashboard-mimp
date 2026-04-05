# Dashboard: Violencia Psicológica contra la Mujer · Perú

## Archivos
- `dashboard.py` — Aplicación principal Dash/Plotly
- `datos_distritos.csv` — Datos sintéticos de 678 distritos
- `peru_distritos.geojson` — Polígonos distritales para el mapa

## Instalación
```bash
pip install dash dash-bootstrap-components plotly pandas numpy geopandas
```

## Ejecución
```bash
python dashboard.py
```
Abre tu navegador en: http://localhost:8050

## Columnas del CSV
| Columna | Descripción |
|---|---|
| district / region | Nombre del distrito y región |
| area | Urbana / Rural |
| pobreza_pct | % pobreza del distrito |
| pobreza_quintil | Q1–Q5 |
| educ_baja_pct | % mujeres con educación baja |
| nivel_educativo | Alto / Medio / Bajo / Muy bajo |
| pct_mujeres_18_35 | % mujeres en grupo etario de mayor riesgo |
| poblacion_mujeres | Población femenina estimada |
| tolerancia_social | Score tolerancia a la violencia (0–1) |
| riesgo | Score de riesgo distrital (0–1) |
| acceso | Score de acceso a servicios (0–1) |
| prioridad | riesgo × (1 − acceso) |
| n_cem | Número de CEMs en el distrito |
| n_refugios | Número de refugios |
| n_otros_servicios | Otros servicios de apoyo |
| linea100_cobertura | ¿Tiene cobertura Línea 100? |
| acceso_telefonia_pct | % cobertura de telefonía |
| riesgo_cat | Bajo / Medio / Alto / Crítico |
| semaforo | Adecuado / Medio / Crítico |
| recomendacion | Recomendación automática |

## Para usar datos reales
Reemplaza `datos_distritos.csv` con tu propia data (mismas columnas)
y `peru_distritos.geojson` con el shapefile oficial de INEI/IGN.
