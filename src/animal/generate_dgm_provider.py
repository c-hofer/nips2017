import multiprocessing
import scipy.misc
import scipy.ndimage

from src.sharedCode.fileSys import Folder
from src.sharedCode.gui import SimpleProgressCounter
from src.sharedCode.provider import Provider

from ..sharedCode.generate_dgm_provider_shapes import *


def job(args):
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
    src_folder = Folder(data_path)
    class_folders = src_folder.folders()

    n = sum([len(cf.files(name_pred=lambda n: n != 'Thumbs.db')) for cf in class_folders])
    progress = SimpleProgressCounter(n)
    progress.display()

    views = {}
    for i in range(1, number_of_directions + 1):
        views['dim_0_dir_{}'.format(i)] = {}
        views['dim_1_dir_{}'.format(i)] = {}
    job_args = []

    for class_folder in class_folders:
        for view in views.values():
            view[class_folder.name] = {}

        for sample_file in class_folder.files(name_pred=lambda n: n != 'Thumbs.db'):
            args = {'file_path': sample_file.path,
                    'label': class_folder.name,
                    'sample_id': sample_file.name,
                    'number_of_directions': number_of_directions}
            job_args.append(args)

    if n_cores == -1:
        n_cores == int(multiprocessing.cpu_count()*0.5)

    pool = multiprocessing.Pool(n_cores)

    errors = []
    for result in pool.imap(job, job_args):
        try:
            label = result['label']
            sample_id = result['sample_id']

            if 'error' in result:
                errors.append((sample_id, result['error']))
            else:
                for view_id, dgm in result['dgms'].items():
                    views[view_id][label][sample_id] = dgm
            progress.trigger_progress()

        except Exception as ex:
            errors.append(ex)

    prv = Provider()
    for key, view_data in views.items():
        prv.add_view(key, view_data)

    meta = {'number_of_directions': number_of_directions}
    prv.add_meta_data(meta)

    prv.dump_as_h5(output_file_path)

    if len(errors) > 0:
        print(errors)

#TODO beautify
# if __name__ == '__main__':
#     from argparse import ArgumentParser
#     import os.path
#
#     parser = ArgumentParser()
#     parser.add_argument('input_folder_path', type=str)
#     parser.add_argument('output_file_path', type=str)
#     parser.add_argument('number_of_directions', type=int)
#     parser.add_argument('--n_cores', type=int, default=4)
#
#     args = parser.parse_args()
#
#     output_dir = os.path.dirname(args.output_file_path)
#     if not os.path.exists(output_dir):
#         print(output_dir, 'does not exist.')
#     else:
#         generate_dgm_provider(args.input_folder_path, args.output_file_path, args.number_of_directions, n_cores=args.n_cores)
