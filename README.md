# 🏨 Hotel Bookings — Dashboard de Analítica y Visualización

**Alejandro López | Entrega Individual**

Los integrantes que realizaron el siguiente proyecto son:
+ Cristobal Sanchez Oliver
+ Mendoza Cruz Axel Ademar
  
## Temas cubiertos
- Evaluación y procesamiento de datos (outliers, tipos, escalas)
- Muestreo, filtrado y transformación de variables
- Dashboard de visualización (Plotly + Streamlit)
- Proyección lineal: **PCA** (Scree Plot + Biplot)
- Proyecciones no lineales: **t-SNE**, **LLE**, **Mapa de Sammon**
- Correlaciones: **Pearson**, **Spearman**, **Kendall**
- Prueba de **Chi-cuadrado** (Cramér's V)
- Explorador interactivo de variables

## Cómo correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Asegúrate de que `HotelData.csv` esté en la misma carpeta que `app.py`.

## Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub (público o privado).
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Selecciona el repositorio, rama `main`, archivo `app.py`.
4. Sube `HotelData.csv` como archivo del repo (o usa `st.file_uploader`).
5. Click **Deploy** — listo.

## Estructura del proyecto

```
hotel_dashboard/
├── app.py              ← Dashboard principal
├── HotelData.csv       ← Dataset
├── requirements.txt    ← Dependencias
└── README.md           ← Este archivo
```
