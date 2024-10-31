import os
import numpy as np
import json
import xmltodict
import requests

# Create a dictionary which contains IDs (1--400) and (X-Y) of all sensors physically present
# in the Orbit testbed facility. This is later used to produce unique labels for the sensor 
# fingerprinting model.
def generate_node_ids():
    ids = {}
    node_i = 0
    for i in np.arange(1, 21):
        for j in np.arange(1, 21):
            ids[str(i) + "-" + str(j)] = node_i
            node_i = node_i + 1
    return ids

# Uses Orbit API to retrieve information about a given node in the Orbit facility.
# - headers: a dictionary of HTTP headers to be used to retrieve data (can be copied from the Chrome session when visiting the orbit-lab.org website)
# - node_id: node identifier in the format "X-Y" (do NOT add "node" prefix)
# - show: print out node information if true
def get_orbit_node_capabilities(header_authorization, node_id, show = False):
    url = f"https://www.orbit-lab.org/cPanel/status/getNodeCapabilities?node=node{node_id}.grid.orbit-lab.org"

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Authorization": header_authorization,
        "Connection": "keep-alive",
        "Host": "orbit-lab.org",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
        "X-Requested-With": "XMLHttpRequest"
    }

    response = requests.get(url, headers=headers)

    try:
        if response.status_code == 200:
            responseJson = xmltodict.parse(response.text)
            if responseJson['response']['@status'] != "OK":
                print('Device not found')
                return None
            if show: 
                print(json.dumps(responseJson, indent=4))
            return responseJson
        else: return None
    except:
        print(response.text)
        return None

# Saves a Python dictionary to a given JSON file
def save_dict_to_json_file(dictionary, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)

# Retrieves a JSON files and returns in a format of a Python dictionary
def read_json_file_to_dict(file_path):
    with open(file_path, 'r') as json_file:
        dictionary = json.load(json_file)
    return dictionary

# Checks if a given input_string contains any of the allowed_substrings
def contains_allowed_substring(input_string, allowed_substrings):
    for substring in allowed_substrings:
        if substring in input_string: return True
    return False

# Retrieves sensor metadata for a given list of nodes from Orbit API and saves to a JSON file
# - node_list: a list of Orbit node IDs (in a format "X-Y", without "node" prefix)
# - file_path: a path to a JSON file in which the function would store the data
def get_orbit_node_infos(header_authorization, node_list, file_path):    
    node_infos = {}

    for node_id in node_list:
        print(f"Processing: {node_id}", end='')
        node_info = get_orbit_node_capabilities(header_authorization, node_id, show=False)

        if node_info is None:
            print(' 🤷‍♂️')
        else:
            devices = node_infos[node_id] = node_info['response']['action']['devices']
            if devices: node_infos[node_id] = devices['device']
            print(' ✅')
        
    save_dict_to_json_file(node_infos, file_path)
    return node_infos

# Filters the list of available Orbit nodes by availability of specified WiFi chipsets
# - node_infos: dictionary of all available Orbit nodes (with all of their metadata)
# - chip_models: WiFi chipset models (i.e., 5212, 9220, etc)
#
# Additional info:
#     Paper mentions that they were using Atheros 5212, 9220, 9280, and 9580 WiFi cards
#     We need to find the largest number of nodes (for which we have sufficient data)
#     with ONE of these cards on board (remember: we need the same hardware vendor for 
#     better model performance)
#
#     After some experimentation, turns out that 5212 card is most common (47 devices w 500 frame limit)
#
#     Additionally, card 5212 has one device. 
# 
#     Also, uniqueness of the vendor/model can be identified using the @INV_dev_id field.
def filter_nodes_by_wifi_chip(node_infos, chip_models, verbose=False):
    node_list_filtered = []
    for node_id in node_infos:
        node_info = node_infos[node_id]
        if not node_info: # some nodes are null
            continue

        node_fit_devices = 0
        for device in node_info:
            device_type = device.get("@INV_dev_type")
            device_id = device.get('@INV_dev_id')
            device_name = device.get('@name')
        
            if device_type and contains_allowed_substring(device_type, chip_models):
                if verbose: 
                    print('[', device_id, ']:', node_id, ':', device_name, '(', device_type, ')')
                node_fit_devices = node_fit_devices + 1

        if node_fit_devices > 0:
            node_list_filtered.append(node_id)

    if verbose: print('Nodes with the WiFi card(s) found:', len(node_list_filtered))

    return node_list_filtered

# Filters the list of nodes based on which of them are present in a dataset directory.
# - node_list: list of nodes in format "X-Y" (without "node" prefix)
# - dir_path: path to the directory in which we will be looking for node datasets
# 
# In the directory, we expect to see files named as follows: "packets_nodeX-Y.mat"
def filter_nodes_by_dir_presence(node_list, dir_path, verbose=False):
    dir_nodes = [fname[12:-4] for fname in os.listdir(dir_path)]

    node_list_filtered = []
    for node_id in node_list:
        if not dir_nodes.__contains__(node_id):
            if verbose: 
                print('Such node is not present in the directory.')
            continue

        node_list_filtered.append(node_id)

    return node_list_filtered

# Filters the list of available Orbit nodes by availability of specified types & models of USRPs
# - node_infos: dictionary of all available Orbit nodes (with all of their metadata)
# - usrp_types: types of USRPs (i.e., USRP2)
# - usrp_models: models of USRPs (i.e., N210)
def filter_nodes_by_usrp_model(node_infos, usrp_types, usrp_models, verbose=False):
    node_list_filtered = []
    for node_id in node_infos:
        node_info = node_infos[node_id]
        if not node_info:
            continue

        node_fit_devices = 0
        for device in node_info:
            device_type = device.get("@INV_dev_type")
            device_id = device.get('@INV_dev_id')
            device_name = device.get('@name')
            device_motherboard = device.get('@INV_mother_board_type')
        
            if device_type and device_motherboard and contains_allowed_substring(device_type, usrp_types) and contains_allowed_substring(device_motherboard, usrp_models):
                if verbose: 
                    print('[', device_id, ']:', node_id, ':', device_name, '(', device_type, ' | ', device_motherboard, ')')
                node_fit_devices = node_fit_devices + 1

        if node_fit_devices > 0:
            node_list_filtered.append(node_id)

    if verbose: print('Nodes with the USRP(s) found:', len(node_list_filtered))

    return node_list_filtered