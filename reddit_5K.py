if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    import time

    # ensure chofer_torchex and pershombox are available
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.path.join(os.path.dirname(__file__)))

    sys.path.append(os.path.join(parent, 'chofer_torchex'))
    sys.path.append(os.path.join(parent, 'tda-toolkit'))

    from src.reddit_5K.generate_dgm_provider import generate_dgm_provider
    from src.reddit_5K.experiments import experiment

    from src.sharedCode.data_downloader import download_provider, download_raw_data
    from src.sharedCode.gui import ask_user_for_provider_or_data_set_download

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/reddit_5K.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/reddit_multi_5K.graph')

    if not os.path.isfile(provider_path):
        choice = ask_user_for_provider_or_data_set_download()

        if choice == "download_data_set":
            download_raw_data("reddit_5K")
            generate_dgm_provider(raw_data_path,
                                  provider_path)

        elif choice == "download_provider":
            download_provider("reddit_5K")
            time.sleep(1) # included since sometimes downloaded file is not yet available when experiment starts.

    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')

    accuracies = []
    n_runs = 5
    for i in range(1, n_runs + 1):
        print('Start run {}'.format(i))
        result = experiment(provider_path)
        accuracies.append(result)

    with open(os.path.join(os.path.dirname(__file__), 'result_reddit5K.txt'), 'w') as f:
        for i, r in enumerate(accuracies):
            f.write('Run {}: {}\n'.format(i, r))
        f.write('\n')
        f.write('mean: {}\n'.format(np.mean(accuracies)))

