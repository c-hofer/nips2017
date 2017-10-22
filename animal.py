if __name__ == '__main__':
    import os

    from src.animal.generate_dgm_provider import generate_dgm_provider
    from src.animal.experiments import experiment

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/npht_animal_corrected_32dirs.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/animal_corrected')

    if not os.path.isfile(provider_path):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        generate_dgm_provider(raw_data_path,
                              provider_path,
                              32)
    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')
    experiment(provider_path)

