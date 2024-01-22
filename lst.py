import streamlit as st
st.set_page_config(page_title='Flood Classification', layout='wide')
import geemap.foliumap as geemap
import utilities as ut
import subprocess
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from utilities import scoring_metrics, read_accuracy_results
from sklearn.model_selection import StratifiedKFold
import json
import tempfile
from utilities import read_accuracy_results
import os

####### MAIN APPLICATION #######
c1, c2, c3 = st.columns([1, 8, 1])
c2.title('Flood Classification')
c1, c2, c3 = st.columns([1, 8, 1])
c2.markdown(
    """
Malik-2211600255
    """
)

row1_col1, row1_col2 = st.columns([2, 1])

ut.initialize_session_state()

with row1_col1:
    # INITIALIZE MAP
    m = geemap.Map(plugin_Draw=True, draw_export=True, add_google_map=False)
    # m.setCenter(76,22, st.session_state.zoom_level)
    m.add_basemap("HYBRID")

    with row1_col2:
        # # GEOCODER - SEARCH LOCATION
        # ut.add_geocoder(mapObject=m)

        # AOI SELECTION
        ut.add_aoi_selector(mapObject=m)

        if st.session_state.aoi == "Not Selected":
            st.info("Select AOI to Proceed")
        else:
            # # SET PROCESSING PARAMETERS
            sessionState = ut.set_params()

        # If the user has submitted the processing form, then run the algorithm
        if 'FormSubmitter:processing-params-Submit' not in st.session_state:
            st.session_state['FormSubmitter:processing-params-Submit'] = False

        # Check if the form has been submitted
        if st.session_state['FormSubmitter:processing-params-Submit']:
            try:
                # Set Processing Options to Proceed
                st.info("Set Processing Options to Proceed")

                # Run the analyse.py script using subprocess
                subprocess.run(["python", "analyse.py"])

                # Get the accuracy results from the temporary file
                temp_filename = ut.read_accuracy_results()

                if temp_filename is not None:
                    with open(temp_filename, 'r') as temp_file:
                        accuracy_data = json.load(temp_file)

                    # Display analysis results
                    if accuracy_data is not None:
                        accuracy_value = accuracy_data.get('accuracy', None)

                        if accuracy_value is not None:
                            st.write(f'Accuracy from analyse.py: {accuracy_value:.4f}')
                            # Display Random Forest results
                            st.write("Random Forest Results:")
                            st.write("Confusion Matrix:")
                            st.write(accuracy_data.get('confusion_matrix', 'Not available'))
                            st.write("Classification Report:")
                            st.write(accuracy_data.get('classification_report', 'Not available'))
                        else:
                            st.warning('Accuracy not found in the output.')
                    else:
                        st.warning('Error getting accuracy results.')

            except Exception as e:
                st.warning(f"Error running analyse.py: {e}")

# Display accuracy results from JSON file
def show_accuracy_from_json(json_filename):
    # Load accuracy results from JSON file
    with open(json_filename, 'r') as json_file:
        accuracy_results = json.load(json_file)
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            accuracy_results = json.load(json_file)
        # ... (lanjutkan dengan menampilkan hasilnya)
    else:
        st.warning(f'File not found: {json_filename}')    
    if accuracy_results:
        # Display accuracy results
        st.write("Accuracy Results from JSON:")
        st.write("Confusion Matrix:")
        st.write(accuracy_results.get('confusion_matrix', 'Not available'))
        st.write("Classification Report:")
        st.write(accuracy_results.get('classification_report', 'Not available'))
        # Add more details as needed

    else:
        st.warning('Error reading accuracy results from JSON.')

# Display the map in the Streamlit app
m.to_streamlit(height=700, width=1000)

# Display accuracy results from JSON file
json_filename = 'accuracy_result.json'  # Update with your actual JSON filename
show_accuracy_from_json(json_filename)

# Display analysis results
ut.show_results()
