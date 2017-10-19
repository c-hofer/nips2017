if __name__ == '__main__':
    import os

    from multiprocessing import cpu_count

    from src.mpeg7.generate_dgm_provider import generate_dgm_provider
    from src.mpeg7.experiments import experiment

    data_path = os.path.join(os.getcwd(), 'data/dgm_provider/npht_mpeg7_corrected_32dirs.h5')
    if not os.path.isfile(data_path):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        n_cores = max(1, cpu_count() - 1)
        generate_dgm_provider('./data/raw_data/mpeg7_corrected',
                              data_path,
                              32,
                              n_cores=n_cores)
    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')
    experiment(data_path)

