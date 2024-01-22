# Import libraries
import sys
import datetime
import pandas as pd
import numpy as np
import ee
import geemap
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from IPython.display import Image
import streamlit as st
from sklearn.metrics import classification_report
from google.auth import compute_engine
import json
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
from utilities import initialize_sessionState, add_geocoder, uploaded_file_to_gdf, add_aoi_selector, set_params, run_analyse_script
from sklearn.model_selection import train_test_split
from utilities import initialize_sessionState, add_geocoder, uploaded_file_to_gdf, add_aoi_selector, set_params, run_analyse_script
import tempfile
from utilities import save_accuracy_results
import os

# Initialize Earth Engine
# credentials = compute_engine.Credentials(scopes=['https://www.googleapis.com/auth/earthengine'])
# ee.Authenticate()
ee.Initialize(project='ee-malik')

# Import custom utilities
from utilities import initialize_sessionState, add_geocoder, uploaded_file_to_gdf, add_aoi_selector, set_params, run_analyse_script

# Set start and end date
startTime = datetime.datetime(2019, 12, 31)
endTime = datetime.datetime(2020, 4, 30)

# roi
geometry = ee.Geometry.Polygon(
    [
      ee.Geometry.Point([106.68271677723821, -6.37329240848678]),
      ee.Geometry.Point([106.97454111805853, -6.37329240848678]),
      ee.Geometry.Point([106.97454111805853, -6.087287504946566]),
      ee.Geometry.Point([106.68271677723821, -6.087287504946566]),
      ee.Geometry.Point([106.68271677723821, -6.37329240848678])
    ]
)

image = ee.ImageCollection('LANDSAT/LC08/C02/T1') \
    .filterDate(startTime, endTime) \
    .filterBounds(geometry)

collection = image.first()

# Filter the Landsat image collection
collection = ee.ImageCollection('LANDSAT/LC08/C02/T1') \
    .filterDate(startTime, endTime) \
    .filterBounds(geometry) \
    .sort('CLOUD_COVER') \
    .first()

# Clip the image according to the geometry
clipped_image = collection.clip(geometry)

# Get the Map ID
map_id = clipped_image.getMapId({
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 30000,
    'gamma': 1.4
})

# Create a folium map object
my_map = folium.Map(location=[-6.23, 106.75], zoom_start=12)

# Add the Earth Engine layer to the folium map
folium.TileLayer(
    tiles=map_id['tile_fetcher'].url_format,
    attr='Google Earth Engine',
    overlay=True,
    name='Landsat Image'
).add_to(my_map)

import folium
from IPython.display import display

# Calculate MNDWI
mndwi = clipped_image.normalizedDifference(['B3', 'B6']).rename('MNDWI')

# Define threshold values for flood classification
threshold_ranges = [-1, 0, 0.2, 0.33, 1]
threshold_palette = ['yellow', 'blue', 'darkblue']

# Initialize an empty water mask
water_mask = ee.Image(0)

# Create water masks based on threshold ranges
for i in range(len(threshold_ranges) - 1):
    current_mask = mndwi.gte(threshold_ranges[i]).And(mndwi.lt(threshold_ranges[i + 1]))
    water_mask = water_mask.where(current_mask, i + 1)

# Clip the water mask to the selected geometry
water_mask = water_mask.clip(geometry)

# Display the water mask on the map
map_id_water_mask = water_mask.getMapId({
    'palette': threshold_palette,
    'min': 0,
    'max': len(threshold_ranges) - 0.6
})

# Create a folium map object
my_map = folium.Map(location=[-6.23, 106.75], zoom_start=10)

# Add the water mask layer to the folium map
folium.TileLayer(
    tiles=map_id_water_mask['tile_fetcher'].url_format,
    attr='Google Earth Engine',
    overlay=True,
    name='Water Mask'
).add_to(my_map)

collection.getInfo()

# sys.maxsize
info = image.getRegion(geometry, 500).getInfo()

print(info[0]);
sys.maxsize

SWIR = image.select("B6")
GREEN = image.select("B3")

# MNDWI = (GREEN - Shortwave infrared 1) / (GREEN + Shortwave infrared 1)

# Reshape image collection
header = info[0]

header

datasets = pd.DataFrame(info, columns=header)

datasets

datasets = datasets.drop(0)

datasets

datasets = datasets.dropna()
datasets

datasets = datasets.drop(columns=['QA_RADSAT'])

DateTime = []
for i in datasets['time']:
    DateTime.append(datetime.datetime.fromtimestamp(i / 1000))

datasets['time'] = DateTime

datasets['time']
datasets['time'].head()

datasets.head()

"""preprocessing"""

# reshape data
reshapeData = np.array(datasets[1:])

# List of used image bands
band_list = ['B3', u'B6']

iBands = [header.index(b) for b in band_list]

# print(reshapeData)
yData = reshapeData[0:, iBands].astype(float)

yData

GREEN = yData[:, 0]
SWIR = yData[:, 1]

MNDWI = (GREEN - SWIR) / (GREEN + SWIR)

MNDWI

datasets = datasets.drop(2)

datasets['MNDWI'] = MNDWI

datasets

datasets.describe()

# minmax scaler untuk scaling data
minmaxScaler = MinMaxScaler()
datasets['MNDWI_SCALE'] = minmaxScaler.fit_transform(datasets[['MNDWI']])

datasets

# make class
datasets.loc[(datasets['MNDWI'] < 0), 'Class'] = 1
datasets.loc[(datasets['MNDWI'] > 0) & (datasets['MNDWI'] < 0.2), 'Class'] = 2
datasets.loc[(datasets['MNDWI'] > 0.2) & (datasets['MNDWI'] < 0.4), 'Class'] = 3
datasets.loc[(datasets['MNDWI'] > 0.4) & (datasets['MNDWI'] < 1), 'Class'] = 4

# make label
datasets.loc[(datasets['MNDWI'] < -0), 'Label'] = 'Non Banjir'
datasets.loc[(datasets['MNDWI'] > 0) & (datasets['MNDWI'] < 0.2), 'Label'] = 'Banjir Ringan'
datasets.loc[(datasets['MNDWI'] > 0.2) & (datasets['MNDWI'] < 0.4), 'Label'] = 'Banjir Sedang'
datasets.loc[(datasets['MNDWI'] > 0.4) & (datasets['MNDWI'] < 1), 'Label'] = 'Banjir Tinggi'

datasets

datasets.describe()

datasets.columns
# pilih data yg sesuai

# training
X = datasets.iloc[:, [4, 5, 6, 7, 8, 9, 10, 11]]
y = datasets.iloc[:, 23]  # label
# pilih data yg sesuai [1,2,4,5,6,7,8,10,11,12,13,14,15,16,17,18]]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# **Modeling - RFC**

# Definisikan metrik evaluasi yang diinginkan
scoring_metrics = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted',
    'cohen_kappa': make_scorer(cohen_kappa_score)
}

# Lakukan cross-validation dan hitung nilai rata-rata untuk setiap metrik
results = perform_analysis(X, y)

for metric_name, result in results.items():
    print(f'{metric_name}: {result.mean():.4f} (std: {result.std():.4f})')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# rubah data training

model = RandomForestClassifier(n_estimators=1000, random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Values:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa_score = cohen_kappa_score(y_test, y_pred)

# Print the Kappa score
print("Cohen's Kappa Score:", kappa_score)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

st.text(f'Random Forest Accuracy: {accuracy:.4f}')

# Stastistikal Analisis
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mape = np.mean(np.abs((y_test - y_pred) / np.maximum(1, y_test))) * 100
r2 = r2_score(y_test, y_pred)
ave = np.var(y_pred / np.var(y_test))

print("R-squared (R2) Score:", r2)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Percentage Error:", mape)
print("Mean Squared Error:", mse)
print("Average Variance Extracted (AVE)", ave)

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test.astype('int'), y_pred.astype('int'))))
print("Classification Report: \n {}\n".format(classification_report(y_test.astype('int'), y_pred.astype('int'))))
print("Accuracy_score: {0:.4f}\n".format(accuracy_score(y_test.astype('int'), y_pred.astype('int')) * 100))
print("Kappa Score: {0:.4f}\n".format(cohen_kappa_score(y_test.astype('int'), y_pred.astype('int'))))


def save_accuracy_results(accuracy, precision, recall, f1, kappa_score):
    # Menghasilkan dictionary dengan hasil akurasi
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cohen_kappa': kappa_score
    }

    # Menyimpan hasil akurasi dalam file JSON
    with open('accuracy_result.json', 'w') as json_file:
        json.dump(results, json_file)


save_accuracy_results(accuracy, precision, recall, f1, kappa_score)

def save_accuracy_results(accuracy, precision, recall, f1, kappa_score):
    # Generate a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cohen_kappa': kappa_score
        }

        # Save accuracy results in the temporary file
        json.dump(results, temp_file)

        # Get the name of the temporary file
        temp_filename = temp_file.name

    # Return the name of the temporary file
    return temp_filename

def read_accuracy_results(file_path='accuracy_result.json'):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                accuracy_results = json.load(json_file)
            return accuracy_results
        else:
            st.warning(f"The file '{file_path}' does not exist.")
            return None
    except Exception as e:
        st.warning(f"Error reading accuracy results from JSON: {e}")
        return None