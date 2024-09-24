import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import matplotlib.pyplot as plt
from datetime import datetime 
import glob
import json 
import os 
from pathlib import Path 
import re 
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from joblib import dump
import joblib
from joblib import load
#from ydata_profiling import ProfileReport
#from fancyimpute import IterativeImputer

st.set_page_config(
    page_title="Dengue Ciencia de Datos",
    page_icon=":bar_chart:",
    layout="centered",
)

st.image("dengue_img.JPEG", use_column_width=True) # Mostrar la imagen de portada

st.title("Proyecto BootCamp de Ciencia de Datos.")
st.markdown("<h3 style='font-size: 20px;'>[Código Facilito]</h3>", unsafe_allow_html=True)

# Sección de introducción
st.write("En este Web App se presentarán los hallazgos del proyecto perteneciente a **Eduardo García Trejo**")
st.write(
    """
    Bienvenidos a continuación se describe el tema a desarrollar: 

    El proyecto se basa en un set de datos de las ciudades de San Juan, Puerto Rico e Iquitos, Perú. El propósito de los datos es poder predecir los casos de dengue que pueden llegar a suceder en el futuro 

    Por lo tanto la pregunta a resolver es:  *¿Cuántos casos de dengue se presentarán en los siguientes años en dichas ciudades?*
    """
)

df = pd.read_csv('dengue_features_train.csv')
df_labels_train = pd.read_csv('dengue_labels_train.csv')
df_casos = df_labels_train['total_cases']
df = pd.concat([df, df_casos], axis=1)
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

st.write("En la siguente tabla se presenta una **muestra** los datos con los que se trabajarón:")
st.dataframe(df.sample(5))


#__________________________________________________________________________________________________

#ANALISIS EXPLORATIORIO DE DATOS 
st.markdown("<h2 style='font-size: 30px;'>Análisis Exploratorio de Datos (EDA)</h2>", unsafe_allow_html=True)

st.write('El DataFrame tiene', len(df), 'filas y', len(df.columns), 'Columnas ', '\n')
st.markdown("<h3 style='font-size: 20px;'>Renombar columnas:</h3>", unsafe_allow_html=True)
st.write('Se realiza un cambio de nombre de las columnas del dataframe original, ya que no son tan fáciles de recordar;  Ejemplo se cambia: **station_max_temp_c** por **Maximum temperature**.')

#Rename Columns 
columns_names ={  
                'city' : 'City',
                'year' : 'Year',
                'weekofyear' : 'Week of year',
                'week_start_date' : 'Week Start Date',
                'station_max_temp_c' : 'Maximum temperature',
                'station_min_temp_c' : 'Minimum temperature',
                'station_avg_temp_c' : 'Average temperature',
                'station_precip_mm' : 'Total precipitation (GHCN)', #Global Historical Climatology Network
                'station_diur_temp_rng_c' : 'Diurnal temperature range(GHCN)',
                'precipitation_amt_mm' : 'Total precipitation (PERSIANN)', #Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks
                'reanalysis_sat_precip_amt_mm' : 'Total precipitation (NCEP)', #National Centers for Environmental Prediction
                'reanalysis_dew_point_temp_k' : 'Mean dew point temperature',
                'reanalysis_air_temp_k' : 'Mean air temperature',
                'reanalysis_relative_humidity_percent' : 'Mean relative humidity',
                'reanalysis_specific_humidity_g_per_kg' : 'Mean specific humidity',
                'reanalysis_precip_amt_kg_per_m2' : 'Total precipitation (NCEP in kg/m2)',
                'reanalysis_max_air_temp_k' : 'Maximum air temperature',
                'reanalysis_min_air_temp_k' : 'Minimum air temperature',
                'reanalysis_avg_temp_k' : 'Average air temperature',
                'reanalysis_tdtr_k' : 'Diurnal temperature range (NCEP)',
                'ndvi_se' : 'Pixel southeast of city centroid',
                'ndvi_sw' : 'Pixel southwest of city centroid',
                'ndvi_ne' : 'Pixel northeast of city centroid',
                'ndvi_nw' : 'Pixel northwest of city centroid'}
st.write("Quedando como resultado: ")
df = df.rename(columns = columns_names)
st.write(df.columns, '\n')

st.write('Mostramos la información del DataFrame tal como número de valores, máximos, mínimos, media, desviación estandar etc.')
st.write(df.describe())

st.markdown("<h3 style='font-size: 20px;'>Valores Duplicados</h3>", unsafe_allow_html=True)
st.write('Se realiza la consulta en busca de valores duplicados y se observa que no hay: ',df[df.duplicated()])

st.markdown("<h3 style='font-size: 20px;'>Revisando Datos Nulos</h3>", unsafe_allow_html=True)
st.write('Tamaño del DataFrame con valores nulos: ', df.index.size)
st.write('Tamaño del DataFrame con **SIN** valores nulos: ', len(df.dropna()))
st.write('**Con lo anterior se observa que si hay valores nulos**, procedemos a graficar los valores nulos')

plt.figure(figsize=(10, 6))
st.image("15 df_nan", use_column_width=False)

st.pyplot(plt)

dif = df.index.size - len(df.dropna())
pct_val_na = (dif * 100 ) / df.index.size
st.write('Se calcula que el porcentaje de valores nulos es: ', f'{pct_val_na:.2f}', '% en base a la formula:' )  
st.write('( (N total filas - N sin valores nulos)*100 ) / N total filas')
st.write(pd.isnull(df).sum())

st.write(''' 
        Se observa que una parte de datos faltantes corresponden a la semana 53, por lo cual se investiga si esos años tuvieron 53 semanas
        se encuentra que solo el año 2005 cuenta con 53 semanas por lo que se toma la decisión de eliminar esas filas del DataFrame 
        ''')

df.loc[df['Week of year']== 53]
copia_df = df.copy()
copia_df.drop(copia_df.loc[copia_df['Week of year']== 53].index, inplace=True) #Se eliminan 5 filas 
copia_df.drop(copia_df[copia_df['Mean air temperature'].isnull()].index, inplace=True) 

st.write(copia_df.isnull().sum())

st.write('Para las columnas restantes se usaran metodos de imputación para rellenar los datos faltantes, para el caso de las columnas **Pixel northeast of city centroid, Pixel northwest of city centroid, Pixel southeast of city centroid y Pixel southwest of city centroid** se utiliza imputación por el vecino mas cercano (KNN), y para el resto del DataFrame se apoya de la imputación multiple. ')

#divide el Data frame en 3, df_cardinal_points para imputar con el vecino mas cercano y df_rest imputación multiple 
df_city = copia_df.iloc[:,:4]
df_pixels = copia_df.iloc[:,4:8]
df_variables = copia_df.iloc[:,8:]

# Crear el imputador KNN y pasamos el dataframe de los puntos cardinales 
imputer = KNNImputer(n_neighbors=2)  
df_imputed = imputer.fit_transform(df_pixels) # Entrena e imputar los datos
df_pixels_imputed = pd.DataFrame(df_imputed, columns=df_pixels.columns)
df_pixels_imputed = df_pixels_imputed.reset_index(drop=True)
st.write(df_pixels_imputed.isnull().sum())

#Rellena las volumnas con las varibles, con imputación multiple 

imputer = IterativeImputer(max_iter=10, random_state=0) # Crear el imputador IterativeImputer
variables_imputed = imputer.fit_transform(df_variables) # Ajustar e imputar los datos
df_variables_imputed = pd.DataFrame(variables_imputed, columns=df_variables.columns) # Convertir el resultado de nuevo a un DataFrame

#concatenan los 3 dataframes
df_city = df_city.reset_index(drop=True)
df_pixels_imputed = df_pixels_imputed.reset_index(drop=True)
df_variables_imputed = df_variables_imputed.reset_index(drop=True)

df_cleaned = pd.concat([df_city, df_pixels_imputed, df_variables_imputed], axis=1) #Se concatenan los 3 DataFrames 

st.write('Posterior a las imputaciones se observan el DataFrame sin valores nulos: ')
plt.figure(figsize=(10, 6))
st.image("16 df_no_nan", use_column_width=False)
st.pyplot(plt)


#__________________________________________________________________________________________________

#Visualización Efectiva
st.markdown("<h2 style='font-size: 30px;'>Visualización Efectiva</h2>", unsafe_allow_html=True)
st.write('En esta sección se presentarán diversos graficos para entender más a fondo todos los valores de cada característica de las que está conformado el DataFrame')

#Graficas de Análisis de normalidad de los datos 
st.write('**Graficas de Análisis de normalidad de los datos**')
def plot_distribution(df, column_name):
    # Crear un DataFrame para la columna seleccionada
    df_col = df[[column_name]]
    
    # Histograma
    hist = alt.Chart(df_col).mark_bar(size=5).encode(
        alt.X(f'{column_name}:Q', bin=alt.Bin(maxbins=30), title='Valor'),
        alt.Y('count():Q', title='Frecuencia'),
        #color=alt.Color(f'{column_name}:Q')
        
    ).properties(
        title=f'Histograma de {column_name}',
        width=150,
        height=150,
    )
    
    return hist 
def chunk_list(lst, chunk_size):
    """Divide una lista en sublistas de tamaño chunk_size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
df_chart = df_cleaned.iloc[:,4:]
# Crear una lista para almacenar los gráficos
charts = []
# Iterar sobre las columnas y generar los gráficos
for column in df_chart.columns:
    chart = plot_distribution(df_chart, column)
    charts.append(chart)
# Dividir los gráficos en grupos de 5
groups = chunk_list(charts, 3)
# Combinar cada grupo horizontalmente
horizontal_charts = [alt.hconcat(*group) for group in groups]
# Combinar todos los grupos verticalmente
combined_chart = alt.vconcat(*horizontal_charts).resolve_scale(
    y='shared'
)
st.altair_chart(combined_chart, use_container_width=True)

#heatmap
matriz_correlacion = df.iloc[:,1:].corr(method='pearson').round(4)
matriz_long = matriz_correlacion.stack().reset_index()
matriz_long.columns = ['columna 1', 'columna 2', 'Correlaciones']

heatmap = alt.Chart(matriz_long).mark_rect().encode(
    x='columna 2:O',
    y='columna 1:O', 
    color=alt.Color("Correlaciones:Q", scale=alt.Scale(scheme='lightgreyred')),
    tooltip=['columna 1', 'columna 2', 'Correlaciones']
).properties(
    width=700,
    height=700,
    title='Mapa de Correlación de Variables'
)
st.altair_chart(heatmap, use_container_width=False)
st.write("De acuerdo al gráfico de correlación de las variables se puede validar que las variables **Mean dew point temperature y Mean specific humidity** tienen una fuerte correlación, por lo que se realiza una grafíca de correlación entre ambas:")

#correlación entre varibles  Histograma para Maximum temperature y Mean specific humidity
grafico_linea = alt.Chart(df_cleaned).mark_point().encode(
    x=alt.X('Mean dew point temperature:Q', title='Temperatura media del punto de rocío (°K)', scale=alt.Scale(domain=[288, 300])),
    y=alt.Y('Mean specific humidity:Q',  title='Humedad específica media', scale=alt.Scale(domain=[10, 22])),
    tooltip=[
            alt.Tooltip('Mean dew point temperature:Q', title='Temperatura', format=',.1f'),  # Con comas y un decimal
            alt.Tooltip('Mean specific humidity:Q', title='Humedad', format='.1f')  # Entero
    ]
).properties(
    title='Relación entre Temperatura media del punto de rocío y Humedad específica media',
    width=300,
    height=300
)

regression_chart = alt.Chart(df_cleaned).mark_point().encode(
    x=alt.X('Mean dew point temperature:Q',  title='Temperatura media del punto de rocío', scale=alt.Scale(domain=[288, 300])),
    y=alt.Y('Mean specific humidity:Q', title='Humedad específica media', scale=alt.Scale(domain=[10, 22])),
    tooltip=['Mean dew point temperature:O', 'Mean specific humidity:Q']
).properties(
    width=300,
    height=300
).transform_regression(
    'Mean dew point temperature', 'Mean specific humidity'
).mark_line()
chart_variables = grafico_linea | regression_chart
st.altair_chart(chart_variables, use_container_width=False)

# Histograma para las temperaturas
st.write('¿Cuál es la distribución de las temperaturas máximas, mínimas y promedio en el conjunto de datos?')
# Histograma para Maximum temperature
histograma1 = alt.Chart(df_cleaned).mark_bar(color='orange').encode(
    alt.X('Maximum temperature:Q', bin=alt.Bin(maxbins=20), title='Maximum temperature'),
    alt.Y('count():Q', title='Cantidad'),
    tooltip=[alt.Tooltip('Maximum temperature:Q', title='Max Temp'), alt.Tooltip('count():Q', title='Cantidad')]
).properties(
    title='Histogramas de las temperaturas Máximas, Promedio y Mínimas',
    width=600,
    height=400
)
# Histograma para Average temperature
histograma2 = alt.Chart(df_cleaned).mark_bar(color='red').encode(
    alt.X('Average temperature:Q', bin=alt.Bin(maxbins=20), title='Average temperature'),
    alt.Y('count():Q', title='Cantidad'),
    tooltip=[alt.Tooltip('Average temperature:Q', title='Avg Temp'), alt.Tooltip('count():Q', title='Cantidad')]
).properties(
    title='Distribución de las temperaturas promedio',
    width=600,
    height=400
)
# Histograma para Minimum temperature
histograma3 = alt.Chart(df_cleaned).mark_bar(color='lightcoral').encode(
    alt.X('Minimum temperature:Q', bin=alt.Bin(maxbins=20), title='Minimum temperature'),
    alt.Y('count():Q', title='Cantidad'),
    tooltip=[alt.Tooltip('Minimum temperature:Q', title='Min Temp'), alt.Tooltip('count():Q', title='Cantidad')]
).properties(
    title='Distribución de las temperaturas mínimas',
    width=600,
    height=400
)
chart = histograma1 + histograma2 + histograma3
st.altair_chart(chart, use_container_width=True)
st.write('Apartir del gráfico anterior se puede observar que las temperaturas que mas se repiten son 23.9 como minima y 32.8 como máxima, además se observa que las temperaturas promedio son menos frecuentes.')

# Gráfico de temperatura promedio por semana
st.write('')
average_temp = df_cleaned.groupby('Week of year')['Average temperature'].mean().reset_index()
average_temp = pd.DataFrame(average_temp)
line_chart = alt.Chart(average_temp).mark_line().encode(
    x=alt.X('Week of year:O', title='Semana del Año'),
    y=alt.Y('Average temperature:Q', title='Temperatura Promedio', scale=alt.Scale(domain=[25, 29])),
    tooltip=['Week of year:O','Average temperature:Q']
).properties(
    title='Temperatura Promedio durante el Año (por semana)',
    width=600,
    height=400
)
st.altair_chart(line_chart, use_container_width=True)
st.write('''En base al grafíco de linea mostrado encontramos que la temperatura promedio menor se encuentra en la **semana 6 con 25.8 °C** y la temperatura promedio mayor se aprecia en la **semana 37 con 28.3 °C**
        por lo que el rango promedio calculado es de 2.5 °C''')

#Temperatura del aíre en ambas ciudades
st.write('')
st.write('**Temperatura del aíre en ambas ciudades**')
st.write('En el siguiente gráfico se presentan las temperaturas del aíre mínimas y máximas por cada mes en las dos ciudades.')
df_sj = df_cleaned[df_cleaned['City'] == 'sj']
df_iq = df_cleaned[df_cleaned['City'] != 'sj']

df_air_temp_sj =df_sj[['Week Start Date', 'Maximum air temperature', 'Minimum air temperature', 'Mean air temperature']]
df_air_temp_sj['Maximum air temperature'] = df_air_temp_sj['Maximum air temperature'] - 273.15
df_air_temp_sj['Minimum air temperature'] = df_air_temp_sj['Minimum air temperature'] - 273.15
df_air_temp_sj['Mean air temperature'] = df_air_temp_sj['Mean air temperature'] - 273.15

df_air_temp_iq =df_iq[['Week Start Date', 'Maximum air temperature', 'Minimum air temperature', 'Mean air temperature']]
df_air_temp_iq['Maximum air temperature'] = df_air_temp_iq['Maximum air temperature'] - 273.15
df_air_temp_iq['Minimum air temperature'] = df_air_temp_iq['Minimum air temperature'] - 273.15
df_air_temp_iq['Mean air temperature'] = df_air_temp_iq['Mean air temperature'] - 273.15

#Grafico san juan, pto rico
bar = alt.Chart(df_air_temp_sj).mark_bar(cornerRadius=10, height=10).encode(
    x=alt.X('min(Minimum air temperature):Q').scale(domain=[16, 34]).title('Temperature (°C)'),
    x2='max(Maximum air temperature):Q',
    color=alt.Color('Mean air temperature:Q', scale=alt.Scale(domain=[20, 30], range=['#ffff00', '#ffeb00', '#ffd700', '#ffb600', '#ff9e00', '#ff8700', '#ff6e00', '#ff5700', '#ff3f00', '#ff0000'])).title(None),
    y=alt.Y('month(Week Start Date):O').title(None),
    tooltip=[
        alt.Tooltip('month(Week Start Date):O', title='Mes'),  # Añadir título a la columna de tooltip
        alt.Tooltip('min(Minimum air temperature):Q', title='Temp Min ', format='.1f'),
        alt.Tooltip('max(Maximum air temperature):Q', title='Temp Max', format='.1f') 
    ]
)

text_min = alt.Chart(df_air_temp_sj).mark_text(align='right', dx=-5).encode(
    x='min(Minimum air temperature):Q',
    y=alt.Y('month(Week Start Date):O'),
    text='min(Minimum air temperature):Q'
)

text_max = alt.Chart(df_air_temp_sj).mark_text(align='left', dx=5).encode(
    x='max(Maximum air temperature):Q',
    y=alt.Y('month(Week Start Date):O'),
    text='max(Maximum air temperature):Q'
)

df_air_temp_sj_chart = (bar + text_min + text_max).properties(
    title=alt.Title(text='Variación de Temperatura del Aire por Mes', subtitle='San Juan, Puerto Rico de 1990 al 2008')
)

#Iquitos, Peru
bar = alt.Chart(df_air_temp_iq).mark_bar(cornerRadius=10, height=10).encode(
    x=alt.X('min(Minimum air temperature):Q').scale(domain=[10, 45]).title('Temperature (°C)'),
    x2='max(Maximum air temperature):Q',
    color=alt.Color('Mean air temperature:Q', scale=alt.Scale(domain=[15, 30], range=['#ffff00', '#ffeb00', '#ffd700', '#ffb600', '#ff9e00', '#ff8700', '#ff6e00', '#ff5700', '#ff3f00', '#ff0000'])).title(None),
    y=alt.Y('month(Week Start Date):O').title(None),
    tooltip=[
        alt.Tooltip('month(Week Start Date):O', title='Mes'),  # Añadir título a la columna de tooltip
        alt.Tooltip('min(Minimum air temperature):Q', title='Temp Min ', format='.1f'),
        alt.Tooltip('max(Maximum air temperature):Q', title='Temp Max', format='.1f') 
    ]
)

text_min = alt.Chart(df_air_temp_iq).mark_text(align='right', dx=-3).encode(
    x='min(Minimum air temperature):Q',
    y=alt.Y('month(Week Start Date):O'),
    text='min(Minimum air temperature):Q'
)

text_max = alt.Chart(df_air_temp_iq).mark_text(align='left', dx= 3).encode(
    x='max(Maximum air temperature):Q',
    y=alt.Y('month(Week Start Date):O'),
    text='max(Maximum air temperature):Q'
)

df_air_temp_iq_chart = (bar + text_min + text_max).properties(
    title=alt.Title(text='Variación de Temperatura del Aire por Mes', subtitle='Iquitos, Perú de 2000 al 2010')
)

chart_tem_aire = df_air_temp_sj_chart | df_air_temp_iq_chart
st.altair_chart(chart_tem_aire, use_container_width=True)

#Precipitaciones en los 3 sistemas de monitoreo 
st.write('''
        Para realizar un analisis de los datos respecto los diferentes sistemas de monitoreo primeramente se describen 
        las diversas fuentes de donde se obtuvieron los datos:
        
        - Global Historical Climatology Network - (GHCN)
        - National Centers for Environmental Prediction - (NCEP)
        - Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks - (PERSIANN)
''')
st.write('')
st.write('¿Existen diferencias significativas entre las precipitaciones reportadas por PERSIANN, NCEP y GHCN?')
# Transformar el DataFrame a formato largo para Altair
df_long = df_cleaned.melt(
    value_vars=[
        'Total precipitation (PERSIANN)', 
        'Total precipitation (NCEP)', 
        'Total precipitation (GHCN)'
    ],
    var_name='Source',
    value_name='Precipitation'
)
# Crea una paleta de colores
colors = alt.Scale(domain=['Total precipitation (PERSIANN)', 'Total precipitation (NCEP)', 'Total precipitation (GHCN)'], range=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Puedes cambiar estos colores
# Crear el gráfico de caja
box_plot = alt.Chart(df_long).mark_boxplot(size=100).encode(
    x=alt.X('Source:N', title='Fuente de Precipitación', sort=None),
    y=alt.Y('Precipitation:Q', title='Precipitación Total'),
    color=alt.Color('Source:N', scale=colors),
    tooltip=['Source:N', 'Precipitation:Q']
).properties(
    title='Diferencias entre las precipitaciones reportadas (PERSIANN, NCEP y GHCN)',
    width=600,
    height=400
).configure_axisX(
    labelAngle=0 
)
st.altair_chart(box_plot, use_container_width=False)
st.write('Se observa que los datos de los sitemas PERSIANN y NCEP son iguales ya que se observan los mismos valores en ambas cajas, por lo que podría ser una buena opcíon descartar una columna para manejar menos caracteristicas.')

#Grafico de precipitaciones y temperaturas por estaciónes del año.
st.write('**Diferencia en precipitaciones y temperaturas entre estaciones del año**')
st.write('En los siguientes gráficos de cajas se muestran los datos de las precipitaciones tomadas por los diferentes sistemas de monitoreo y las temperaturas Máximas, Mínimas y Promedio, divididos por las 4 estaciones del año.')

copy_clean = df_cleaned.copy()
copy_clean['season'] = df_cleaned['Week of year'].apply(lambda x: 'Winter' if x in [49, 50, 51, 52, 1, 2, 3] else 'Spring' if x in [4, 5, 6, 7, 8, 9] else 'Summer' if x in [10, 11, 12, 13, 14, 15] else 'Fall')

Prec_por_estacion1 = alt.Chart(copy_clean).mark_boxplot(size=80).encode(
    x=alt.X('season', title=''),
    y=alt.Y('Total precipitation (PERSIANN)', title='Precipitación Total (PERSIANN)', axis=alt.Axis(titleAnchor='middle')),
    color=alt.Color('season:N', legend=None),
    tooltip=['season:N', 'Total precipitation (PERSIANN):Q']
).properties(
    width=390,
    height=200,
    title='Diferencia en precipitaciones por cada estación del año'
)

Prec_por_estacion2 = alt.Chart(copy_clean).mark_boxplot(size=80).encode(
    x=alt.X('season', title=''),
    y=alt.Y('Total precipitation (GHCN)', title='Precipitación Total (GHCN)'),
    color=alt.Color('season:N', legend=None),
    tooltip=['season:N', 'Total precipitation (GHCN):Q']
).properties(
    width=390,
    height=200,
)

Prec_por_estacion3 = alt.Chart(copy_clean).mark_boxplot(size=80).encode(
    x=alt.X('season', title=''),
    y=alt.Y('Total precipitation (NCEP)', title='Precipitación Total (NCEP)'),
    color=alt.Color('season:N', legend=None),
    tooltip=['season:N', 'Total precipitation (NCEP):Q']
).properties(
    width=390,
    height=200,
)

precipitaciones =  Prec_por_estacion1 & Prec_por_estacion2 & Prec_por_estacion3

temp_por_estacion1 = alt.Chart(copy_clean).mark_boxplot(size=80).encode(
    x=alt.X('season', title=''),
    y=alt.Y('Maximum temperature', title='Temperatura Maxíma', scale=alt.Scale(domain=[10, 45])),
    color=alt.Color('season:N', legend=None),
    tooltip=['season:N', 'Maximum temperature:Q']
).properties(
    width=390,
    height=200,
    title='Diferencia en Temperaturas por cada estación del año'
)

temp_por_estacion2 = alt.Chart(copy_clean).mark_boxplot(size=80).encode(
    x=alt.X('season', title=''),
    y=alt.Y('Average temperature', title='Temperatura Promedio', scale=alt.Scale(domain=[10, 45])),
    color=alt.Color('season:N', legend=None),
    tooltip=['season:N', 'Average temperature:Q']
).properties(
    width=390,
    height=200,
)

temp_por_estacion3 = alt.Chart(copy_clean).mark_boxplot(size=80).encode(
    x=alt.X('season', title=''),
    y=alt.Y('Minimum temperature', title='Temperatura Minima', scale=alt.Scale(domain=[10, 45])),
    color=alt.Color('season:N', legend=None),
    tooltip=['season:N', 'Minimum temperature:Q']
).properties(
    width=390,
    height=200,
)

temperaturas = temp_por_estacion1 & temp_por_estacion2 & temp_por_estacion3
boxes = alt.hconcat(precipitaciones, temperaturas)
st.altair_chart(boxes, use_container_width=True)
st.write('''De la visualización anterior se puede comparar que en el otoño es cuando las temperaturas son más altas, 
        y comparando con las precipitaciones podemos notar que  hay mayor cantidad de Outliers. 
        Por otra parte en primavera los datos de las temperaturas tienen una  menor variabilidad y el mismo fenómeno se observa para las gráficas de precipitación.''')

#NDVI
st.write('')
st.write('**¿Qué datos interesantes se pueden obtener del índice de vegetación de diferencia normalizada (NDVI)?**')
st.write(''' 
        Con el **(NDVI)** Es posible identificar la salud de la vegetación 
        alrededor de la ciudad en las 4 dir. cardinales, o entender cómo varía la vegetación en el entorno urbano. 
- (1) vegetación densa y saludable, 
- (0) áreas vegetación escasa o sin vegetación, 
- (Negativos)  cuerpos de agua o áreas no vegetadas

En el entendido que las columnas de interés son las siguientes de acuerdo a la relación que se tiene previo al cambio de nombre de las columnas. 

- ndvi_se – Pixel southeast of city centroid
- ndvi_sw – Pixel southwest of city centroid
- ndvi_ne – Pixel northeast of city centroid
- ndvi_nw – Pixel northwest of city centroid''')

#Promedio de NDVI por año en las cuatro regiones 
avg_ndvi_sj =df_sj.groupby('Year')[['Pixel northwest of city centroid', 'Pixel northeast of city centroid',  'Pixel southwest of city centroid', 'Pixel southeast of city centroid']].mean()
avg_ndvi_sj = avg_ndvi_sj.reset_index()

avg_ndvi_iq =df_iq.groupby('Year')[['Pixel northwest of city centroid', 'Pixel northeast of city centroid',  'Pixel southwest of city centroid', 'Pixel southeast of city centroid']].mean()
avg_ndvi_iq = avg_ndvi_iq.reset_index()

#Se transforma la tabla para garficar los datos 
df_melted_sj = pd.melt(avg_ndvi_sj, id_vars='Year' ,var_name='NDVI')
df_melted_iq = pd.melt(avg_ndvi_iq, id_vars='Year' ,var_name='NDVI')

#Grafico
ndvi_sj_chart = alt.Chart(df_melted_sj).mark_area().encode(
    x=alt.X("Year:O", title=''),
    y=alt.Y("value:Q", title=''),
    color="NDVI:N",
    row=alt.Row("NDVI:N").sort(['Pixel northwest of city centroid', 'Pixel northeast of city centroid',  'Pixel southwest of city centroid', 'Pixel southeast of city centroid'],
    axis=alt.Axis(labels=False)
    ),
    tooltip=["Year:O", "value:Q", "NDVI:N"]
).properties(height=100, 
            width=500,
            title='Distribución promedio de NDVI por año en San Juan'
)

ndvi_iq_chart = alt.Chart(df_melted_iq).mark_area().encode(
    x=alt.X("Year:O", title=''),
    y=alt.Y("value:Q", title=''),
    color="NDVI:N",
    row=alt.Row("NDVI:N").sort(['Pixel northwest of city centroid', 
                                'Pixel northeast of city centroid',  
                                'Pixel southwest of city centroid', 
                                'Pixel southeast of city centroid'], axis=alt.Axis(labels=False)
    ), 
    tooltip=["Year:O", "value:Q", "NDVI:N"]
).properties(height=100, 
            width=500,
            title='Distribución promedio de NDVI por año en Iquitos'
)
ndvi_por_cd = alt.hconcat(ndvi_sj_chart, ndvi_iq_chart)
st.altair_chart(ndvi_por_cd, use_container_width=True)

st.write('')
st.write('Con el análisis del gráfico podemos deducir que el promedio de vegetación se redujo drásticamente en la parte norte de la ciudad de San Juan a partir del año 2003, caso contrario en Iquitos que se mantuvo constante tanto en el norte como en el sur.')

#Análisis de la humedad relativa media 
st.write('')
st.write('**¿Cúal es la relación de la humedad relativa media a través de los años dividido en trimestres?**')
columnas = ['City', 'Year', 'Week of year', 'Week Start Date', 'Mean relative humidity']
relative_humedity = df_cleaned[columnas]

humed_rel_sj = relative_humedity[relative_humedity['City'] == 'sj']
humed_rel_iq = relative_humedity[relative_humedity['City'] == 'iq']

#Clasifica por trimestres
def get_quadrimester(month):
    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [ 7, 8, 9]:
        return 3
    elif month in [10, 11, 12]:
        return 4
    else:
        return None  # Por si acaso hay valores inesperados
# Aplicar la función para obtener trimestres
humed_rel_sj['Trimestre'] = humed_rel_sj['Week Start Date'].dt.month.apply(get_quadrimester)
humed_rel_iq['Trimestre'] = humed_rel_iq['Week Start Date'].dt.month.apply(get_quadrimester)

humed_por_q_sj  = humed_rel_sj.groupby(['Year', 'Trimestre'])['Mean relative humidity'].mean().reset_index()
humed_por_q_iq =  humed_rel_iq.groupby(['Year', 'Trimestre'])['Mean relative humidity'].mean().reset_index()

#Humedad de Sj
# Size of the dropbins
size = 20
# Count of distinct x features
xFeaturesCount = 4
# Count of distinct y features
yFeaturesCount = 25
drop_shape = "M0,0 C-0.5,1 -1,1.5 -1,2 C-1,2.5 -0.5,3 0,3 C0.5,3 1,2.5 1,2 C1,1.5 0.5,1 0,0 Z"

color_scale = alt.Scale(domain=[73.35, 82.75], range=['#add8e6', '#b0d0e1', '#93b7d4', '#78a1c7', '#6492c1', '#4a7bb4', '#3464a7', '#2a4f8f', '#1f3d6b', '#00008b'])
chart1 = alt.Chart(humed_por_q_sj).mark_point(size=size**2, shape=drop_shape).encode(
    alt.X('xFeaturePos:Q')
        .title('Trimestre')
        .axis(labelPadding=20, grid=False, tickOpacity=0, domainOpacity=0),
    alt.Y('Year:N', title='Year')
        .title('Year')
        .axis(grid=True, labelPadding=0, tickOpacity=0, domainOpacity=0),
    stroke=alt.value('black'),
    strokeWidth=alt.value(2),
    fill=alt.Fill('mean(Mean relative humidity):Q').scale(color_scale).legend(title='Humedad Relativa', orient='right'),
    tooltip=['Year:N', 'Trimestre:N', 'Mean relative humidity:Q']
).transform_calculate(
    # Calcular la posición X en base al cuatrimestre (3 cuatrimestres por fila)
    xFeaturePos='datum.Trimestre'
).properties(
    # Exact scaling factors to make the hexbins fit
    width=size * xFeaturesCount * 1.2,
    height=size * yFeaturesCount * 1.6,  
    title=alt.Title(text='Humedad Relativa Media por Trimestre', subtitle='San Juan de 1990 al 2008')
).configure_view(
    strokeWidth=0
)

#Humedad iq
color_scale = alt.Scale(domain=[75, 95.30], range=['#add8e6', '#9ac3d0', '#6fa8b3', '#478da5', '#1d7598', '#005f8e', '#004d7a', '#003e6b', '#002f5c', '#00008b'])
chart2 =  alt.Chart(humed_por_q_iq).mark_point(size=size**2, shape=drop_shape).encode(
    alt.X('xFeaturePos:Q')
        .title('Trimestre')
        .axis(labelPadding=20, grid=False, tickOpacity=0, domainOpacity=0),
    alt.Y('Year:N', title='Year')
        .title('Year')
        .axis(grid=True, labelPadding=0, tickOpacity=0, domainOpacity=0),
    stroke=alt.value('black'),
    strokeWidth=alt.value(2),
    fill=alt.Fill('mean(Mean relative humidity):Q').scale(color_scale).legend(title='Humedad Relativa', orient='right'),
    tooltip=['Year:N', 'Trimestre:N', 'Mean relative humidity:Q']
).transform_calculate(
    # Calcular la posición X en base al cuatrimestre (3 cuatrimestres por fila)
    xFeaturePos='datum.Trimestre'
).properties(
    # Exact scaling factors to make the hexbins fit
    width=size * xFeaturesCount * 1.2,
    height=size * yFeaturesCount*1.1 ,  
    title=alt.Title(text='Humedad Relativa Media por Trimestre', subtitle='Iquitos, Perú de 2000 al 2010')
).configure_view(
    strokeWidth=0
)

# Usar columnas para mostrar los gráficos uno al lado del otro
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(chart1, use_container_width=True)
with col2:
    st.altair_chart(chart2, use_container_width=True)

st.write('''Derivado a la interpretación del gráfico de la izquierda podemos deducir que en la ciudad de San Juan durante el primer trimestre la humedad relativa media es menor en comparación al resto del año, otra tendencia notable es que del 1990 al 2000 la humedad era mayor a comparación de del 2000 al 2008.

De lado derecho los valores indican que en Iquitos los primeros dos trimestres de los años son más húmedos en comparación con los  últimos dos trimestres; Siendo el tercer trimestre el menos húmedo. 
''')

#Años con mas casos en ambas ciudades 
st.write('')
st.write('**¿Cúal es el año con mayor casos de dengue en las dos ciudades de estudio?**')
#Casos maximos por año
casos_sj = df_cleaned[df_cleaned['City'] == 'sj']
casos_sj = casos_sj.groupby(['Year'])['total_cases'].sum().reset_index().sort_values(by='total_cases', ascending=False)
casos_iq = df_cleaned[df_cleaned['City'] == 'iq']
casos_iq = casos_iq.groupby(['Year'])['total_cases'].sum().reset_index().sort_values(by='total_cases', ascending=False)
col1, col2 = st.columns(2)
with col1:
    st.write('San Juan', casos_sj.head(), use_container_width=True)
with col2:
    st.write('Iquitos',casos_iq.head(), use_container_width=True)

st.write('Se observa que los años con más casos son 1994 y 2008 para las ciudades de San Juan e Iquitos respectivamente')
casos_sj_1994 = df_cleaned[(df_cleaned['Year'] == 1994) & (df_cleaned['City'] == 'sj')]
casos_iq_2008 = df_cleaned[(df_cleaned['Year'] == 2008) & (df_cleaned['City'] == 'iq')]

st.write('En base a la informción anterior se crea un gráfico de los casos por cada semana, teniendo como resultado lo siguiente: ')
#Casos San Juan, 1994 
color_scale = alt.Scale(domain=[13, 461.0], range=['#FFFF00', '#FFEB00', '#FFD700', '#FFB600', '#FF9E00', '#FF8700', '#FF6E00', '#FF5700', '#FF3F00', '#FF0000'])
casos_sj_chart = alt.Chart(casos_sj_1994).mark_rect().encode(
    alt.X("date(Week Start Date):O").title("Día").axis(format="%e", labelAngle=0),
    alt.Y("month(Week Start Date):O").title("Mes"),
    alt.Color("total_cases").title(None).scale(color_scale).legend(title='Casos'),
    tooltip=[
        alt.Tooltip("monthdate(Week Start Date)", title="Date"),
        alt.Tooltip("total_cases", title="Total de Casos"),
    ],
).configure_view(
    step=25,
    strokeWidth=0
).configure_axis(
    domain=False
).properties(title=alt.Title(text='Número de Casos por Semana', subtitle='San Juan, Puerto Rico. 1994'))
st.altair_chart(casos_sj_chart, use_container_width=True)

#Casos Iquitos 2008
color_scale = alt.Scale(domain=[0, 63], range=['#FFFF00', '#FFEB00', '#FFD700', '#FFB600', '#FF9E00', '#FF8700', '#FF6E00', '#FF5700', '#FF3F00', '#FF0000'])
casos_iq_chart = alt.Chart(casos_iq_2008).mark_rect().encode(
    alt.X("date(Week Start Date):O").title("Día").axis(format="%e", labelAngle=0),
    alt.Y("month(Week Start Date):O").title("Mes"),
    alt.Color("total_cases").title(None).scale(color_scale).legend(title='Casos'),
    tooltip=[
        alt.Tooltip("monthdate(Week Start Date)", title="Date"),
        alt.Tooltip("total_cases", title="Total de Casos"),
    ],
).configure_view(
    step=25,
    strokeWidth=0
).configure_axis(
    domain=False
).properties(title=alt.Title(text='Número de Casos por Semana', subtitle='Iquitos, Perú. 2008'))

st.altair_chart(casos_iq_chart, use_container_width=True)
st.write('''
En las visualizaciones anteriores es claro que en San Juan en 1994 se presentaron más casos en los últimos meses del año, siendo octubre y noviembre los que tienen mayor concentración de casos.

Respecto a la ciudad de Iquitos es un tanto similar pues se tiene una mayor concurrencia en Octubre y Septiembre, Además de esos meses, se notan mayor incidencia en Enero, Febrero y la primera semana de marzo.
''')


st.write('**Haciendo un análisis de las tendencias de los casos a través de todos los años en el set de datos.**')
# Casos máximos por año
casos_year_and_month = df_cleaned[['City', 'Year', 'total_cases']].copy()
casos_year_and_month['Month'] = df_cleaned['Week Start Date'].dt.month
casos_sj_year_m = casos_year_and_month[df_cleaned['City'] == 'sj']
casos_sj_year_m = casos_sj_year_m.groupby(['Year','Month'])['total_cases'].sum().reset_index().sort_values(by=['Year', 'Month'], ascending=True)

casos_p_anio = casos_sj_year_m.groupby('Year')['total_cases'].sum().reset_index().sort_values(by='Year', ascending=True)

# Primer gráfico
colors_green = ['#E0FFD4', '#B3F8A2', '#8CEB7D', '#66DE58', '#40D13A', '#2BB628', '#1E9D1B', '#138F14', '#0D7A0E', '#006600']
color_scale_m = alt.Scale(domain=[0, 2000], range=colors_green)

casos_sj_year_m_chart = alt.Chart(casos_sj_year_m).mark_rect().encode(
    alt.X("Year:O").title("").axis(labelAngle=90, labelPadding=20),
    alt.Y("Month:O").title("Mes"),
    alt.Color("total_cases").title(None).scale(color_scale_m).legend(title='Casos'),
    tooltip=[
        alt.Tooltip("Month", title="Mes"),
        alt.Tooltip("total_cases", title="Total de Casos"),
    ],
).properties(title=alt.Title(text='Número de Casos en cada año y mes', subtitle='San Juan, Puerto Rico'))

# Segundo gráfico
casos_sj_year_chart = alt.Chart(casos_p_anio).mark_rect(fill='#138F14').encode(
    alt.X("Year:O").title("").axis(labelExpr="null", domain=False, ticks=False),
    alt.Y("total_cases:Q").title("Casos por Año").scale(reverse=True).axis(labelExpr="null", domain=False, ticks=False, grid=False),
    tooltip=[
        alt.Tooltip("total_cases", title="Total de Casos"),
    ]
)
# Concatenar los gráficos verticalmente y aplicar configuración
combined_chart = alt.vconcat(
    casos_sj_year_m_chart,
    casos_sj_year_chart
).configure_view(
    step=17,
    strokeWidth=0
).configure_axis(
    domain=False
)

casos_iq_year_m = casos_year_and_month[df_cleaned['City'] == 'iq']
casos_iq_year_m = casos_iq_year_m.groupby(['Year','Month'])['total_cases'].sum().reset_index().sort_values(by=['Year', 'Month'], ascending=True)

casos_p_anio_iq = casos_iq_year_m.groupby('Year')['total_cases'].sum().reset_index().sort_values(by='Year', ascending=True)

# Primer gráfico
colors_green = ['#E0FFD4', '#B3F8A2', '#8CEB7D', '#66DE58', '#40D13A', '#2BB628', '#1E9D1B', '#138F14', '#0D7A0E', '#006600']
color_scale_a = alt.Scale(domain=[0, 250], range=colors_green)

casos_iq_year_m_chart = alt.Chart(casos_iq_year_m).mark_rect().encode(
    alt.X("Year:O").title("").axis(labelAngle=90, labelPadding=20),
    alt.Y("Month:O").title("Mes"),
    alt.Color("total_cases").title(None).scale(color_scale_a).legend(title='Casos'),
    tooltip=[
        alt.Tooltip("Month", title="Mes"),
        alt.Tooltip("total_cases", title="Total de Casos"),
    ],
).properties(title=alt.Title(text='Número de Casos en cada año y mes', subtitle='Iquitos, Perú'))

# Segundo gráfico
casos_iq_year_chart = alt.Chart(casos_p_anio_iq).mark_rect(fill='#138F14').encode(
    alt.X("Year:O").title("").axis(labelExpr="null", domain=False, ticks=False),
    alt.Y("total_cases:Q").title("Casos por Año").scale(reverse=True).axis(labelExpr="null", domain=False, ticks=False, grid=False),
    alt.Color("total_cases").title(None).scale(color_scale_a).legend(title='Casos'),
    tooltip=[
        alt.Tooltip("total_cases", title="Total de Casos"),
    ],
)

# Concatenar los gráficos verticalmente y aplicar configuración
combined_chart2 = alt.vconcat(
    casos_iq_year_m_chart,
    casos_iq_year_chart
).configure_view(
    step=17,
    strokeWidth=0
).configure_axis(
    domain=False
)

# Mostrar el gráfico organizado por columnas
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(combined_chart, use_container_width=True)
with col2:
    st.altair_chart(combined_chart2, use_container_width=True)

st.write('''
Tal como los hallazgos en las gráficas casos por semana, podemos ver que tiene tendencias similares, respecto a San Juan,
se observa mayo incidencia en los últimos cuatro meses de los años, 
        
En la ciudad de Iquitos, igualmente se observa ligeramente una mayor carga de casos en el último cuatrimestre y un poco en los primeros bimestres.''')

st.markdown("<h2 style='font-size: 30px;'>Modelado de Datos</h2>", unsafe_allow_html=True)
st.write('')
st.write('''
        Para el modelado de datos se decide trabajar con **GradientBoostingRegressor**, pues es al ser un modelo de regresión es útil para predecir valores continuos, lo cual es el
        objetivo de este proyecto, otra ventaja del modelo es que al contar con muchos modelos de árbol es más robusto para clasificar los datos, y finalmente proporciona métricas de importancia de características, 
        lo que permite identificar cuáles variables son más relevantes para la predicción. 
        ''')
st.write('')
st.write('Primeramente se separa la columna de total_cases pues es el target y no es conveniente que se le pase al modelo de Machine Learning')
st.image("1. drop_target.jpg", use_column_width=False)
st.write('Se realiza la separación de los datos, en un 80% los datos de entrenamiento y un 20% los datos de validación')
st.image("2.división_df.jpg", use_column_width=True)
st.write('Se crea pipeline de transformación con los transformadores Binarizer para la columna "City y un Escalador para los datos que presentan Outliers')
col1, col2 , col3= st.columns(3)
with col1:
    st.image("3 tranformador.jpg", use_column_width=True)
with col2:
    st.image("4 escalador.jpg", use_column_width=True)
with col3:
    st.image("5 pass.jpg", use_column_width=True)
st.image("6 pipeline completo.jpg", use_column_width=True)
st.image("7 nuevo pipeline.jpg", use_column_width=True)

st.markdown("<h2 style='font-size: 30px;'>Entrenando Modelo</h2>", unsafe_allow_html=True)
st.write('Se entrena el modelo GradientBoostingRegressor con mil arboles')
st.image("8 modelo.jpg", use_column_width=True)

st.markdown("<h2 style='font-size: 30px;'>Validación del Modelo</h2>", unsafe_allow_html=True)
st.write('''
Posterior al entrenamiento resta validar el modelo por lo cual se pasa el set de validación al método predict, 
con esto el modelo hará las predicciones con los nuevos valores que se le pasan, dichas predicciones las recibiran los metodos que nos permiten 
calcular el error cuadrático medio y el coeficiente de determinación R^2 esto con el fin de evaluar el rendimiento del modelo. 
Pues ambos métodos evalúan las predicciones realizadas con los valores que se tienen.''')
st.image("9 validacion.jpg", use_column_width=True)
st.image("10 score.jpg", use_column_width=True)
st.write('Finalmente nos apoyamos del método feature_importances_ para poder visualizar qué porcentaje de importancia tiene cada característica en el modelo que se implementó')
st.image("11 features.jpg", use_column_width=True)
st.image("12 features_rank.jpg", use_column_width=False)

st.markdown("<h2 style='font-size: 30px;'>Persistencia del Modelo</h2>", unsafe_allow_html=True)
st.write('''Para asegurar que el modelo que se entrenó y validó sea reusable es necesario correr el siguiente código, el cúal generará un archivo el cual se podrá cargar en cualquier proyecto como si fuera un objeto,
        mismo que es capaz de hacer predicciones de nuevos set de datos de los cuales se desee hacer una predicción.''')
st.image("13 persistencia.jpg", use_column_width=True)

st.markdown("<h2 style='font-size: 30px;'>Prueba final del Modelo</h2>", unsafe_allow_html=True)
st.write('Se realiza la carga del modelo en el presente proyecto para poder realizar la prueba y hacer nuevas predicciones')

#carga el pipeline con el modelo entrenado previamente
ultimate_inference_pipeline = load('inference_pipeline_proyecto_CF.joblib')
st.image("14 carga.jpg", use_column_width=True)

st.write('Se obtiene el DataFrame de prueba que también se encontraba disponible, con el objetivo de probar el modelo')
df_test = pd.read_csv('features_test_clean.csv')
df_test = df_test.rename(columns = columns_names)
st.dataframe(df_test.sample(5))
st.write('Fecha Menor: ', df_test['Week Start Date'].min())
st.write('Fecha Mayor: ', df_test['Week Start Date'].max())

df_test['Casos Predichos'] = ultimate_inference_pipeline.predict(df_test) # se predice sobre el nuevo set de datos y se guarda la salida en una nueva columna 
df_test['Casos Predichos'] = np.maximum(df_test['Casos Predichos'], 0)
df_test['Casos Predichos'] = np.round(df_test['Casos Predichos']).astype(int)
st.write('Casos que se han predecido con el modelo: ', df_test['Casos Predichos'])
df_final = df_test[['City', 'Week of year', 'Week Start Date', 'Casos Predichos']].sort_values(by='Casos Predichos', ascending=False)
st.write('De esta manera podemos ver las fechas en que posiblemente habría mayor número de casos de Dengue: ', df_final)

st.markdown("<h2 style='font-size: 30px;'>Conclusión</h2>", unsafe_allow_html=True)
st.write('''En base al presente proyecto desarrollado se puede concluir que los datos 
        tienen el suficiente poder predictivo para el problema planteado inicialmente 
        pues una vez evaluado el modelo se obtuvo una confiabilidad del 70%.
        Además  de ello las visualizaciones creadas ayudan a identificar patrones de comportamiento 
        en las características con las que se trabajaron, tal como Temperatura, Precipitaciones, Humedad, etc.
''')










