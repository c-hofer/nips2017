if __name__ == '__main__':
    import os
    import numpy as np

    from src.reddit_12K.generate_dgm_provider import generate_dgm_provider
    from src.reddit_12K.experiments import experiment

    from src.sharedCode.data_downloader import download_provider, download_raw_data
    from src.sharedCode.gui import ask_user_for_provider_or_data_set_download

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/reddit_12K.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/reddit_subreddit_10K.graph')

    if not os.path.isfile(provider_path):

        choice = ask_user_for_provider_or_data_set_download()

        if choice == "download_data_set":
            download_raw_data("reddit_12K")
            generate_dgm_provider(raw_data_path,
                                  provider_path)

        elif choice == "download_provider":
            download_provider('reddit_12K')

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
