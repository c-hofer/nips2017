import tempfile
import zipfile
import requests
import os
import time

from pathlib import Path


_data_set_name_2_raw_data_google_drive_id = \
    {
        'animal': '0BxHF82gaPzgSNTBTbk8yNG1faVk',
        'mpeg7': '0BxHF82gaPzgSbVRxS0F5ZHh1S2s',
        'reddit_5K': '0BxHF82gaPzgSalA2RnltNFNFOWM',
        'reddit_12K': '0BxHF82gaPzgSRmxFNmlodWpaOUU',

    }


_data_set_name_2_provider_google_drive_id = \
    {
        'animal': '0BxHF82gaPzgSSWIxNmJBRFJzcmM',
        'mpeg7': '0BxHF82gaPzgSU3lPWDNEVHhNR3M',
        'reddit_5K': '0BxHF82gaPzgSZDdFWDU3S29hdm8',
        'reddit_12K': '0BxHF82gaPzgSd0d4WDNYVnN4dEU',
    }


_data_set_name_2_provider_name = \
    {
        'animal': 'npht_animal_32dirs.h5',
        'mpeg7': 'npht_mpeg7_32dirs.h5',
        'reddit_5K': 'reddit_5K.h5',
        'reddit_12K': 'reddit_12K.h5'
    }


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_iter = response.iter_content(CHUNK_SIZE)
        with open(destination, "wb") as f:

            for i, chunk in enumerate(content_iter):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    print(i, end='\r')

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)

    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_raw_data(data_set_name):
    id = _data_set_name_2_raw_data_google_drive_id[data_set_name]

    output_path = os.path.join(str(Path(__file__).parents[2]), 'data/raw_data')

    print('Downloading ... ')
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_f = os.path.join(tmp_dir, 'download')
        download_file_from_google_drive(id, tmp_f)

        with zipfile.ZipFile(tmp_f, 'r') as zip_f:
            zip_f.extractall(output_path)
            time.sleep(1)


def download_provider(data_set_name):
    id = _data_set_name_2_provider_google_drive_id[data_set_name]

    file_name = _data_set_name_2_provider_name[data_set_name]

    output_path = os.path.join(str(Path(__file__).parents[2]), 'data/dgm_provider/{}'.format(file_name))

    print('Downloading ... ')

    ensure_path_existence(output_path)

    download_file_from_google_drive(id, output_path)


def ensure_path_existence(the_path):
    parent_dir = os.path.dirname(the_path)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
