import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pickle
import hashlib
from IPython.display import display

def read_dat_iq_file(file_path):
    raw_data = np.fromfile(file_path, dtype=np.float32)
    
    iq_samples = raw_data[::2] + 1j * raw_data[1::2]
    
    return iq_samples

def request_value_dropdown(prompt, options, callback):
    dropdown = widgets.Dropdown(
        options=options,
        value=options[0],
        description=prompt,
    )

    # Function to handle the dropdown value change
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            print(f"Selected option: {change['new']}")
            callback(change['new'])

    # Attach the function to the dropdown
    dropdown.observe(on_change)

    # Display the dropdown
    display(dropdown)

def generate_grid_node_ids():
    ids = {}
    coordinates = {}
    node_i = 0
    for i in np.arange(1, 21):
        for j in np.arange(1, 21):
            ids[str(i) + "-" + str(j)] = node_i
            node_i = node_i + 1
            coordinates[node_i] = (i, j)
    return ids, coordinates

def apply_ieee_style():
    plt.style.use('default')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
        "axes.linewidth": 1,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1,

        # Grayscale settings
        "image.cmap": "gray",
        "axes.facecolor": 'white',
        "figure.facecolor": 'white',
        "figure.edgecolor": 'white',
        "savefig.facecolor": 'white',
        "savefig.edgecolor": 'white',
        "grid.color": '0.8',
        "text.color": 'black',
        "axes.edgecolor": 'black',
        "axes.labelcolor": 'black',
        "xtick.color": 'black',
        "ytick.color": 'black'
    })

def extract_unix_timestamp_ms(file_path):
    from datetime import datetime
    
    # Extract the timestamp part from the file path
    timestamp_str = file_path.split('_epoch_')[1].replace('.h5', '')
    
    # Convert the timestamp string to a datetime object
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
    
    # Convert the datetime object to unix time in milliseconds
    unix_timestamp_ms = int(timestamp.timestamp() * 1000)
    
    return unix_timestamp_ms

def convert_ms_to_time_label(unix_timestamp_ms):
    # Convert milliseconds to seconds, minutes, and hours
    total_seconds = unix_timestamp_ms // 1000
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    seconds = total_seconds % 60
    
    # Create a time label based on the length of time
    if hours > 0:
        return f'{hours}h'
    elif minutes > 0:
        return f'{minutes}m'
    else:
        return f'{seconds}s'

def hash_object(obj):
    # Serialize the object with pickle
    obj_bytes = pickle.dumps(obj)
    # Hash the serialized data
    hash_object = hashlib.sha256(obj_bytes)
    return hash_object.hexdigest()