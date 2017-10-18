if __name__ == '__main__':
    import os

    from multiprocessing import cpu_count

    from src.animal.generate_dgm_provider import generate_dgm_provider
    from src.animal.experiments import experiment

    if not os.path.isfile('./data/dgm_provider/npht_animal_corrected_32dirs.h5'):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        n_cores = max(1, cpu_count() - 1)
        generate_dgm_provider('./data/raw_data/animal_corrected',
                              './data/dgm_provider/npht_animal_corrected_32dirs.h5',
                              32,
                              n_cores=n_cores)
    else:
        print('Found Persistence diagram provider!')

    print('Starting experiment...')
    experiment()

