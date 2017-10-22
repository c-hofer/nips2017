if __name__ == '__main__':
    import os

    from src.mpeg7.generate_dgm_provider import generate_dgm_provider
    from src.mpeg7.experiments import experiment

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/npht_mpeg7_corrected_32dirs.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/mpeg7_corrected')

    if not os.path.isfile(provider_path):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        generate_dgm_provider(raw_data_path,
                              provider_path,
                              32)
    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')
    experiment(provider_path)

