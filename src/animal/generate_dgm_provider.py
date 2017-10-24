import multiprocessing
import scipy.misc
import scipy.ndimage
import os

from src.sharedCode.fileSys import Folder
from src.sharedCode.gui import SimpleProgressCounter
from src.sharedCode.provider import Provider

from ..sharedCode.generate_dgm_provider_shapes import *


def job(args):
    import warnings
    warnings.filterwarnings('error')#TODO change this ... those errors should not occour

    sample_file_path = args['file_path']
    label = args['label']
    sample_id = args['sample_id']
    number_of_directions = args['number_of_directions']
    return_value = {'label': label, 'sample_id': sample_id, 'dgms': {}}

    img = scipy.misc.imread(sample_file_path, flatten=True)
    img = reduce_to_largest_connected_component(img)
    try:
        npht = get_npht(img, number_of_directions)

    except Exception as ex:
        return_value['error'] = ex
    else:
        dgms_dim_0 = [x[0] for x in npht]
        dgms_dim_1 = [x[1] for x in npht]

        dgms_dim_0 = [threhold_dgm(dgm) for dgm in dgms_dim_0]
        dgms_dim_1 = [threhold_dgm(dgm) for dgm in dgms_dim_1]

        for dir_i, dgm_0, dgm_1 in zip(range(number_of_directions), dgms_dim_0, dgms_dim_1):
            if len(dgm_0) == 0:
                return_value['error'] = 'Degenerate diagram detected.'
                break

            return_value['dgms']['dim_0_dir_{}'.format(dir_i)] = dgm_0
            return_value['dgms']['dim_1_dir_{}'.format(dir_i)] = dgm_1

    return return_value


def generate_dgm_provider(data_path, output_file_path, number_of_directions, n_cores=-1):
    if not os.path.exists(os.path.dirname(output_file_path)):
        print(os.path.dirname(output_file_path), 'does not exist.')

    src_folder = Folder(data_path)
    class_folders = src_folder.folders()

    n = sum([len(cf.files(name_pred=lambda n: n != 'Thumbs.db')) for cf in class_folders])
    progress = SimpleProgressCounter(n)
    progress.display()

    views = {}
    for i in range(0, number_of_directions):
        views['dim_0_dir_{}'.format(i)] = {}
        views['dim_1_dir_{}'.format(i)] = {}
    job_args = []

    for class_folder in class_folders:
        for view in views.values():
            view[class_folder.name] = {}

        for sample_file in class_folder.files(name_pred=lambda name: name != 'Thumbs.db'):
            args = {'file_path': sample_file.path,
                    'label': class_folder.name,
                    'sample_id': sample_file.name,
                    'number_of_directions': number_of_directions}
            job_args.append(args)

    if n_cores == -1:
        n_cores = int(multiprocessing.cpu_count()*0.5)

    with multiprocessing.Pool(n_cores) as pool:

        errors = []
        for i, result in enumerate(pool.imap_unordered(job, job_args)):
            label = result['label']
            sample_id = result['sample_id']

            if 'error' in result:
                errors.append((sample_id, result['error']))
            else:
                for view_id, dgm in result['dgms'].items():
                    views[view_id][label][sample_id] = dgm

            progress.trigger_progress()

    print('Finished calculation ... writing provider (this may also take some time ;) )')
    prv = Provider()
    for key, view_data in views.items():
        prv.add_view(key, view_data)

    meta = {'number_of_directions': number_of_directions}
    prv.add_meta_data(meta)

    prv.dump_as_h5(output_file_path)

    if len(errors) > 0:
        with open('animal_dgm_creation_errors.txt', 'w') as f:
            for k, v in errors:
                f.write(k)
                f.write(str(v))

