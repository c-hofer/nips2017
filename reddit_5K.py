if __name__ == '__main__':
    import os

    from multiprocessing import cpu_count

    from src.reddit_5K.generate_dgm_provider import generate_dgm_provider
    from src.reddit_5K.experiments import experiment

    data_path = os.path.join(os.getcwd(), 'data/dgm_provider/reddit_5k.h5')
    if not os.path.isfile(data_path):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        n_cores = max(1, cpu_count() - 1)
        generate_dgm_provider('./data/raw_data/reddit_5K/reddit_multi_5K.graph',
                              data_path)
    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')
    experiment(data_path)
