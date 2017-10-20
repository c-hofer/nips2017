import pickle
import numpy as np
import functools

from pershombox import toplex_persistence_diagrams
from ..sharedCode.gui import SimpleProgressCounter
from ..sharedCode.provider import Provider


# region general purpose


def load_data(data_set_path):
    with open(data_set_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


class RedditGraph:
    def __init__(self, data):
        self._data = data

    @property
    def vertices(self):
        return list(iter(self._data))

    @property
    def edges(self):
        return_value = set()

        for vertex, vertex_info in self._data.items():
            neighbors = vertex_info['neighbors']
            for neighbor in neighbors:
                return_value.add(tuple(sorted([vertex, neighbor])))

        return list(return_value)

    def vertex_degree(self, vertex_id):
        return len(self._data[vertex_id]['neighbors'])

    def vertex_neighbors(self, vertex_id):
        return self._data[vertex_id]['neighbors']


def norm_dgm(dgm):
    if len(dgm) == 0:
        return np.array([]), np.array([])

    not_essential_points = [p for p in dgm if p[1] != float('inf')]
    essential_points = [p for p in dgm if p[1] == float('inf')]

    mi = min([p[0] for p in dgm])

    ma = None
    if len(not_essential_points) != 0:
        ma = max([p[1] for p in not_essential_points])
    else:
        ma = max([p[0] for p in dgm])

    norm_fact = 1
    if ma != mi:
        norm_fact = ma - mi

    not_essential_points = [[(p[0] - mi) / norm_fact, (p[1] - mi) / norm_fact] for p in not_essential_points]
    essential_points = [[(p[0] - mi) / norm_fact, 1] for p in essential_points]
    return not_essential_points, essential_points


def threhold_dgm(dgm, t):
    return list(p for p in dgm if p[1]-p[0] > t)



def write_provider(output_path, views, labels):
    prv = Provider()

    for key, view in views.items():
        prv.add_view(key, view)

    str_2_int_label_map = {}
    for k in labels:
        str_2_int_label_map[str(int(k))] = int(k)

    prv.add_str_2_int_label_map(str_2_int_label_map)

    prv.dump_as_h5(output_path)


# endregion


class VertexFiltrationBase:
    def __init__(self, reddit_graph):
        self._graph = reddit_graph

    def __call__(self, simplex):
        if type(simplex) == int:
            return self._filtration(simplex)
        else:
            return max([self._filtration(v) for v in simplex])

    @functools.lru_cache(maxsize=None)
    def _filtration(self, vertex):
        return self._filtration_implementation(vertex)

    def _filtration_implementation(self, vertex):
        pass


class DegreeVertexFiltration(VertexFiltrationBase):
    def _filtration_implementation(self, vertex):
        return self._graph.vertex_degree(vertex)


def generate_views_vertex_based_filtrations(data):
    progress = SimpleProgressCounter(len(data['graph']))
    progress.display()

    filtrations = [DegreeVertexFiltration]
    labels = set(data['labels'])

    sub_views = ['_dim_0', '_dim_0_essential', '_dim_1', '_dim_1_essential']
    views = {}
    for f in filtrations:
        name_base = f.__name__
        for s in sub_views:
            views[name_base + s] = {}

    for label in labels:
        for view in views.values():
            view[str(int(label))] = {}

    for id, graph_data in data['graph'].items():

        graph = RedditGraph(graph_data)
        vertices = graph.vertices
        edges = graph.edges
        label = str(int(data['labels'][id]))

        simplices = [(v,) for v in vertices] + edges

        for filt_class in filtrations:
            f = filt_class(graph)

            f_vertices = [f(v) for v in vertices]
            f_edges = [f(e) for e in edges]

            f_values = f_vertices + f_edges

            dgm_0, dgm_1 = toplex_persistence_diagrams(simplices, f_values)

            dgm_0, dgm_0_essential = norm_dgm(dgm_0)
            dgm_1, dgm_1_essential = norm_dgm(dgm_1)

            dgm_0, dgm_0_essential = threhold_dgm(dgm_0, 0.01), threhold_dgm(dgm_0_essential, 0.01)
            dgm_1, dgm_1_essential = threhold_dgm(dgm_1, 0.01), threhold_dgm(dgm_1_essential, 0.01)

            dgm_0, dgm_0_essential = np.array(dgm_0), np.array(dgm_0_essential)
            dgm_1, dgm_1_essential = np.array(dgm_1), np.array(dgm_1_essential)

            f_name = filt_class.__name__

            views[f_name + '_dim_0'][label][str(id)] = dgm_0
            views[f_name + '_dim_0_essential'][label][str(id)] = dgm_0_essential
            views[f_name + '_dim_1'][label][str(id)] = dgm_1
            views[f_name + '_dim_1_essential'][label][str(id)] = dgm_1_essential

        progress.trigger_progress()

    return views, labels


def generate_dgm_provider(input_path, output_path):
    data = load_data(input_path)

    vertex_views, labels = generate_views_vertex_based_filtrations(data)

    views = {**vertex_views}
    write_provider(output_path, views, labels)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os.path

    parser = ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        print(output_dir, 'does not exist.')
    else:
        generate_dgm_provider(args.input_path, args.output_path)
