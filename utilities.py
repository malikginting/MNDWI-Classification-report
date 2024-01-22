import streamlit as st
import ee, datetime
import geemap.foliumap as geemap
import geemap as gmap
import folium
import geopandas as gpd
import subprocess
import json
import os
import subprocess
from datetime import date, timedelta
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report

def initialize_session_state():
    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4
    if st.session_state.get("aoi") is None:
        st.session_state["aoi"] = 'Not Selected'
    if st.session_state.get("useMNDWI") is None:
        st.session_state["useMNDWI"] = True


def uploaded_file_to_gdf(data, crs):
    import tempfile
    import os
    import uuid
    import zipfile

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
        return gdf
    elif file_path.lower().endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(tempfile.gettempdir(), file_id))
        return os.path.join(tempfile.gettempdir(), file_id)
    else:
        gdf = gpd.read_file(file_path, crs=crs, driver='GeoJSON')
        return gdf

def add_aoi_selector(mapObject):
    with st.expander("Select Area of Interest (AOI)", False):
        option = "Upload GeoJSON"
        if option == "Upload GeoJSON":
            uploaded_file = st.file_uploader(
                "Upload a GeoJSON file to use as an AOI.",
                type=["geojson"]
            )
            crs = {"init": "epsg:4326"}

            if uploaded_file is not None:
                gdf = uploaded_file_to_gdf(uploaded_file, crs)
                st.session_state["aoi"] = geemap.geopandas_to_ee(gdf, geodesic=False)
                ee_obj = st.session_state['aoi']

                from shapely.geometry import Polygon, Point

                minx, miny, maxx, maxy = gdf.geometry.total_bounds
                gdf_bounds = gpd.GeoSeries({
                    'geometry': Polygon([Point(minx, maxy), Point(maxx, maxy), Point(maxx, miny), Point(minx, miny)])
                }, crs="EPSG:4326")

                area = gdf_bounds.area.values[0]
                center = gdf_bounds.centroid
                center_lon = float(center.x); center_lat = float(center.y)

                if area > 5:
                    zoom_level = 8
                elif area > 3:
                    zoom_level = 10
                elif 0.1 < area < 0.5:
                    zoom_level = 11
                elif 1 < area < 2:
                    zoom_level = 9
                else:
                    zoom_level = 13

                mapObject.addLayer(ee_obj, {}, 'aoi')
                mapObject.set_center(center_lon, center_lat, zoom_level)
                st.session_state["aoi"] = ee_obj

def save_geojson_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def set_params():
    with st.expander("Define Processing Parameters"):
        form = st.form(key='processing-params')
        from_date = form.date_input('Start Date', datetime.date(2019, 12, 31))
        to_date = form.date_input('End Date', date.today() - timedelta(days=1))
        cloud_cover = form.number_input(label="Cloud Cover Threshold (%)", min_value=0, max_value=50, value=5, step=5)
        satellite = form.selectbox("Landsat Satellite", [
            "Landsat 5",
            "Landsat 7",
            "Landsat 8"
        ], index=2)

        if to_date - from_date < timedelta(days=10):
            st.error('Difference between the two selected data is too small. Try again!')
            st.stop()
        else:
            submit = form.form_submit_button('Submit')

        if submit:
            st.session_state['fromDate'] = from_date
            st.session_state["toDate"] = to_date
            st.session_state["cloudCover"] = cloud_cover
            st.session_state['satellite'] = satellite

            if st.button("Run Analysis"):
                try:
                    run_analyse_script()
                except Exception as e:
                    st.warning(f"Error running analyse.py: {e}")
                else:
                    st.success("Analysis completed successfully!")

def run_analyse_script():
    try:
        subprocess.run(['python', 'analyse.py'], check=True)
    except Exception as e:
        st.warning(f"Error running analyse.py: {e}")
        return False
    else:
        st.success("Analysis completed successfully!")
        return True
def save_accuracy_results(accuracy_data, file_path='accuracy_result.json'):
    """
    Save accuracy results to a JSON file.

    Parameters:
    - accuracy_data (dict): Dictionary containing accuracy results.
    - file_path (str): Path to the JSON file. Default is 'accuracy_result.json'.
    """
    with open(file_path, 'w') as file:
        json.dump(accuracy_data, file)

def read_accuracy_results(json_filename='accuracy_result.json'):
    try:
        with open(json_filename, 'r') as json_file:
            accuracy_results = json.load(json_file)
        return accuracy_results
    except Exception as e:
        print(f"Error reading accuracy results: {e}")
        save_accuracy_results(accuracy_results, json_filename)
        return None
    
def get_accuracy_results(json_filename='accuracy_result.json'):
    try:
        # Membaca hasil akurasi dari file JSON yang dihasilkan oleh analyse.py
        with open(json_filename, 'r') as file:
            accuracy_data = json.load(file)
        return accuracy_data
    except Exception as e:
        st.warning(f"Error getting accuracy results: {e}")
        save_accuracy_results({}, json_filename)
        return None
    
def read_accuracy_results(json_filename):
    try:
        with open(json_filename, 'r') as json_file:
            accuracy_results = json.load(json_file)
        return accuracy_results
    except Exception as e:
        print(f"Error reading accuracy results: {e}")
        return None
    
def show_results():
    accuracy_data = get_accuracy_results()

    if accuracy_data is not None:
        accuracy_value = accuracy_data.get('accuracy', None)

        if accuracy_value is not None:
            st.write(f'Accuracy from analyse.py: {accuracy_value:.4f}')
            # Menampilkan hasil Random Forest
            st.write("Random Forest Results:")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix)
            st.write("Classification Report:")
            st.write(classification_report)

        else:
            st.warning('Accuracy not found in the output.')
    else:
        st.warning('Error getting accuracy results.')

# Definisi scoring_metrics
def scoring_metrics(y_true, y_pred):
    # Implementasi metrik penilaian kustom
    # Contoh: Cohen's Kappa Score
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa

def show(mapObject, state):
    try:
        # Membaca hasil akurasi dari file JSON yang dihasilkan oleh analyse.py
        with open('accuracy_result.json', 'r') as file:
            accuracy_data = json.load(file)

        accuracy_value = accuracy_data.get('accuracy', None)

        if accuracy_value is not None:
            st.write(f'Accuracy from analyse.py: {accuracy_value:.4f}')
        else:
            st.warning('Accuracy not found in the output.')

        # Menampilkan hasil Random Forest
        st.write("Random Forest Results:")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix)
        st.write("Classification Report:")
        st.write(classification_report)

    except Exception as e:
        st.warning(f"Error showing results: {e}")
    else:
        st.info("Processing Options Set. Analysis completed.")

if __name__ == "__main__":
    if st.button("Show Results"):
        if run_analyse_script():
            show_results()
