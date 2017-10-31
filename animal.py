if __name__ == '__main__':
    import os
    import numpy as np

    from src.animal.generate_dgm_provider import generate_dgm_provider
    from src.animal.experiments import experiment
    from src.sharedCode.data_downloader import download_provider, download_raw_data

    provider_path = os.path.join(os.path.dirname(__file__), 'data/dgm_provider/npht_animal_32dirs.h5')
    raw_data_path = os.path.join(os.path.dirname(__file__), 'data/raw_data/animal')

    if not os.path.isfile(provider_path):
        print(\
            """
Persistence diagram provider does not exist! 
You have two options: 
1) Create persistence diagram provider from raw data
   this will take several hours and tda-toollkit has 
   to be installed properly.
2) Download precalculated persistence diagram provider.
   Type 1 or 2 depending on your choice.
            """
        )
        choice = input('-->')
        while str(choice) not in ['1', '2']:
            print("Choice has to be 1 or 2!")
            input()

        if choice == "1":
            # download_raw_data('animal')
            generate_dgm_provider(raw_data_path,
                                  provider_path,
                                  32)

        elif choice == "2":
            download_provider('animal')

    else:
        print('Found persistence diagram provider!')

    print('Starting experiment...')

    accuracies = []
    n_runs = 5
    for i in range(1, n_runs + 1):
        print('Start run {}'.format(i))
        result = experiment(provider_path)
        accuracies.append(result)

    with open(os.path.join(os.path.dirname(__file__), 'result_animal.txt'), 'w') as f:
        for i, r in enumerate(accuracies):
            f.write('Run {}: {}\n'.format(i, r))
        f.write('\n')
        f.write('mean: {}\n'.format(np.mean(accuracies)))

