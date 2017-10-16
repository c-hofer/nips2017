import h5py
import numpy as np


class ProviderError(Exception):
    pass


class NameSpace:
    pass


class Provider:
    _serial_str_keys = NameSpace()
    _serial_str_keys.data_views = 'data_views'
    _serial_str_keys.str_2_int_label_map = 'str_2_int_label_map'
    _serial_str_keys.meta_data = 'meta_data'

    def __init__(self, data_views={}, str_2_int_label_map=None, meta_data={}):
        self.data_views = data_views
        self.str_2_int_label_map = str_2_int_label_map
        self.meta_data = meta_data
        self._cache = NameSpace()

    def add_view(self, name_of_view, view):
        assert type(name_of_view) is str
        assert isinstance(view, dict)
        assert all([type(label) is str for label in view.keys()])
        assert name_of_view not in self.data_views

        self.data_views[name_of_view] = view

    def add_str_2_int_label_map(self, label_map):
        assert isinstance(label_map, dict)
        assert all([type(str_label) is str for str_label in label_map.keys()])
        assert all([type(int_label) is int for int_label in label_map.values()])
        self.str_2_int_label_map = label_map

    def add_meta_data(self, meta_data):
        assert isinstance(meta_data, dict)
        self.meta_data = meta_data

    def _check_views_are_consistent(self):
        if len(self.data_views) > 0:
            first_view = next(iter(self.data_views.values()))

            # Check if every view has the same number of labels.
            lenghts_same = [len(first_view) == len(view) for view in self.data_views.values()]
            if not all(lenghts_same):
                raise ProviderError('Not all views have same amount of label groups.')

            # Check if every view has the same labels.
            labels_same = [set(first_view.keys()) == set(view.keys()) for view in self.data_views.values()]
            if not all(labels_same):
                raise ProviderError('Not all views have the same labels in their label groups.')

            # Check if every label group has the same number of subjects in each view.
            labels = first_view.keys()
            for k in labels:
                label_groups_cons = [set(first_view[k].keys()) == set(view[k].keys()) for view in
                                     self.data_views.values()]
                if not all(label_groups_cons):
                    raise ProviderError('There is some inconsistency in the labelgroups.' \
                                        + ' Not the same subject ids in each view for label {}'.format(k))

    def _check_str_2_int_labelmap(self):
        """
        assumption: _check_views_are_consistent allready called.
        """
        first_view = list(self.data_views.values())[0]

        # Check if the labels are the same.
        if not set(self.str_2_int_label_map.keys()) == set(first_view.keys()):
            raise ProviderError('self.str_2_int_label_map has not the same labels as the data views.')

        # Check if int labels are int.
        if not all([type(v) is int for v in self.str_2_int_label_map.values()]):
            raise ProviderError('Labels in self.str_2_int_label have to be of type int.')

    def _check_state_for_serialization(self):
        if len(self.data_views) == 0:
            raise ProviderError('Provider must have at least one view.')

        self._check_views_are_consistent()

        if self.str_2_int_label_map is not None:
            self._check_str_2_int_labelmap()

    def _prepare_state_for_serialization(self):
        self._check_state_for_serialization()

        if self.str_2_int_label_map is None:
            self.str_2_int_label_map = {}
            first_view = list(self.data_views.values())[0]

            for i, label in enumerate(first_view):
                self.str_2_int_label_map[label] = i + 1

    def dump_as_h5(self, file_path):
        self._prepare_state_for_serialization()

        with h5py.File(file_path, 'w') as file:
            data_views_grp = file.create_group(self._serial_str_keys.data_views)

            for view_name, view in self.data_views.items():
                view_grp = data_views_grp.create_group(view_name)

                for label, label_subjects in view.items():
                    label_grp = view_grp.create_group(label)

                    for subject_id, subject_values in label_subjects.items():
                        label_grp.create_dataset(subject_id, data=subject_values)

            label_map_grp = file.create_group(self._serial_str_keys.str_2_int_label_map)
            for k, v in self.str_2_int_label_map.items():
                # since the lua hdf5 implementation seems to have issues reading scalar values we
                # dump the label as 1 dimensional tuple.
                label_map_grp.create_dataset(k, data=(v,))

            meta_data_group = file.create_group(self._serial_str_keys.meta_data)
            for k, v in self.meta_data.items():
                if type(v) is str:
                    v = np.string_(v)
                    dset = meta_data_group.create_dataset(k, data=v)
                else:
                    meta_data_group.create_dataset(k, data=v)

    def read_from_h5(self, file_path):
        with h5py.File(file_path, 'r') as file:
            # load data_views
            data_views = dict(file[self._serial_str_keys.data_views])
            for view_name, view in data_views.items():
                view = dict(view)
                data_views[view_name] = view

                for label, label_group in view.items():
                    label_group = dict(label_group)
                    view[label] = label_group

                    for subject_id, value in label_group.items():
                        label_group[subject_id] = file[self._serial_str_keys.data_views][view_name][label][subject_id][
                            ()]

            self.data_views = data_views

            # load str_2_int_label_map
            str_2_int_label_map = dict(file[self._serial_str_keys.str_2_int_label_map])
            for str_label, str_to_int in str_2_int_label_map.items():
                str_2_int_label_map[str_label] = str_to_int[()]

            self.str_2_int_label_map = str_2_int_label_map
            for k, v in self.str_2_int_label_map.items():
                self.str_2_int_label_map[k] = int(v[0])

            # load meta_data
            meta_data = dict(file[self._serial_str_keys.meta_data])
            for k, v in meta_data.items():
                meta_data[k] = v[()]
            self.meta_data = meta_data

        return self

    def select_views(self, views: [str]):
        data_views = {}
        for view in views:
            data_views[view] = self.data_views[view]

        return Provider(data_views=data_views, str_2_int_label_map=self.str_2_int_label_map, meta_data=self.meta_data)

    @property
    def sample_id_to_label_map(self):
        if not hasattr(self._cache, 'sample_id_to_label_map'):
            self._cache.sample_id_to_label_map = {}
            for label, label_data in self.data_views[self.view_names[0]].items():
                for sample_id in label_data:
                    self._cache.sample_id_to_label_map[sample_id] = label

        return self._cache.sample_id_to_label_map

    @property
    def view_names(self):
        return list(self.data_views.keys())

    @property
    def labels(self):
        return list(self.data_views[self.view_names[0]].keys())

    @property
    def sample_labels(self):
        for i in range(len(self)):
            _, label = self[i]
            yield label

    @property
    def sample_ids(self):
        if not hasattr(self._cache, 'sample_ids'):
            first_view = self.data_views[self.view_names[0]]
            self._cache.sample_ids = sum([list(label_group.keys()) for label_group in first_view.values()], [])

        return self._cache.sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample_label = self.sample_id_to_label_map[sample_id]

        x = {}

        for view_name, view_data in self.data_views.items():
            x[view_name] = view_data[sample_label][sample_id]

        return x, sample_label

# Example
# """
# from provider import Provider
# import numpy as np
# import h5py
#
# p = Provider()
# view1 = {'label1': {'1': np.random.randn(5), '2': np.random.randn(2)}, 'label2': {'1': np.random.randn(3), '2': np.random.randn(5)}}
# view2 = {'label1': {'1': np.random.randn(5), '2': np.random.randn(2)}, 'label2': {'1': np.random.randn(3), '2': np.random.randn(5)}}
# label_map = {'label1': 1, 'label2': 2}
#
# p.add_view('first_view', view1)
# p.add_view('second', view2)
# p.add_meta_data({'origin': 'this is dummy text.'})
# p.dump_as_h5('/tmp/test.h5')
#
# print('==============================================')
# print('h5 file:')
#
# with h5py.File('/tmp/test.h5', 'r') as f:
#     f.visit(lambda n: print(n))
#
# print('==============================================')
#
# p = Provider()
# p.read_from_h5('/tmp/test.h5')
# print('p.data_views:')
# print(p.data_views)
# print('')
# print('p.str_2_int_label_map:')
# print(p.str_2_int_label_map)
# print('')
# print('p.meta_data :')
# print(p.meta_data)
# """

