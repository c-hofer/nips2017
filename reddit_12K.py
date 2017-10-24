if __name__ == '__main__':
    import os
    import numpy as np

    from src.reddit_12K.generate_dgm_provider import generate_dgm_provider
    from src.reddit_12K.experiments import experiment

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/reddit_12k.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/reddit_12K/reddit_subreddit_10K.graph')

    if not os.path.isfile(provider_path):
        print('Persistence diagram provider does not exists, creating ... (this may need some time)')
        generate_dgm_provider(raw_data_path,
                              provider_path)
    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')

    accuracies = []
    n_runs = 5
    for i in range(1, n_runs + 1):
        print('Start run {}'.format(i))
        result = experiment(provider_path)
        accuracies.append(result)

    with open(os.path.join(os.path.dirname(__file__), 'result_reddit12K.txt'), 'w') as f:
        for i, r in enumerate(accuracies):
            f.write('Run {}: {}\n'.format(i, r))
        f.write('\n')
        f.write('mean: {}\n'.format(np.mean(accuracies)))
