import os
from src.animal.generate_dgm_provider import generate_dgm_provider
from multiprocessing import cpu_count


if os.path.exists('./data/dgm_rovider/animal_corrected.h5'):
    print('Persistence diagram provider does not exists, creating ... (this may need some time)')
    n_cores = max(1, cpu_count()-1)
    generate_dgm_provider('./data/raw_data/animal_corrected',
                          './data/dgm_provider/animal_corrected.h5',
                          32,
                          n_cores=n_cores)
else:
    print('Found Persistence diagram provider!')


