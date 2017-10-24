import scipy.misc
import scipy.ndimage
import multiprocessing

from collections import defaultdict
from ..sharedCode.fileSys import Folder
from ..sharedCode.gui import SimpleProgressCounter
from ..sharedCode.provider import Provider

from ..sharedCode.generate_dgm_provider_shapes import *


def job(args):
    sample_file_path = args['sample_file_path']
    label = args['label']
    sample_id = args['sample_id']
    number_of_directions = args['number_of_directions']
    return_value = {'label': label, 'sample_id': sample_id, 'views': {}}

    img = scipy.misc.imread(sample_file_path, flatten=True)
    img = reduce_to_largest_connected_component(img)

    npht = get_npht(img, number_of_directions)

    dgms_dim_0 = [x[0] for x in npht]
    dgms_dim_1 = [x[1] for x in npht]

    dgms_dim_0 = [threhold_dgm(dgm) for dgm in dgms_dim_0]
    dgms_dim_1 = [threhold_dgm(dgm) for dgm in dgms_dim_1]

    views = return_value['views']

    for i, dgm in enumerate(dgms_dim_0):
        views['dim_0_dir_{}'.format(i)] = dgm

    for i, dgm in enumerate(dgms_dim_1):
        views['dim_1_dir_{}'.format(i)] = dgm

    return return_value


def generate_dgm_provider(data_path, output_path, number_of_directions, n_cores=-1):
    src_folder = Folder(data_path)
    files = src_folder.files(name_pred=lambda n: n.endswith('.gif'))

    progress = SimpleProgressCounter(len(files))
    progress.display()

    job_args = []

    for sample_file in files:
        args = {}

        label = sample_file.name.split('-')[0]
        args['label'] = label
        args['sample_file_path'] = sample_file.path
        args['sample_id'] = sample_file.name
        args['number_of_directions'] = number_of_directions

        job_args.append(args)

    if n_cores == -1:
        n_cores = int(multiprocessing.cpu_count()*0.5)

    views = defaultdict(lambda: defaultdict(dict))
    errors = []

    with multiprocessing.Pool(n_cores) as pool:

        for result in pool.imap(job, job_args):
            try:
                label = result['label']
                sample_id = result['sample_id']

                for view_name, dgm in result['views'].items():
                    views[view_name][label][sample_id] = dgm

                progress.trigger_progress()

            except Exception as ex:
                errors.append(ex)

    prv = Provider()

    for view_name, view in views.items():
        prv.add_view(view_name, view)

    meta = {'number_of_directions': number_of_directions}
    prv.add_meta_data(meta)

    prv.dump_as_h5(output_path)

    if len(errors) > 1:
        print(errors)

