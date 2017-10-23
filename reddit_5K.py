if __name__ == '__main__':
    import os

    from src.reddit_5K.generate_dgm_provider import generate_dgm_provider
    from src.reddit_5K.experiments import experiment

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/reddit_5k.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/reddit_5K/reddit_multi_5K.graph')

    if not os.path.isfile(provider_path):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        generate_dgm_provider(raw_data_path,
                              provider_path)
    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')
    experiment(provider_path)
