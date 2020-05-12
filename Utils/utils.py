# !/usr/local/bin/python

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np


def one_hot_string(map):
    values = np.array(map)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def get_action_from_encoding(enc):
    idx = _actions.index(enc)
    return action_map[idx], idx


####################### GRAPH #######################


def createTemporalDict(in_dict):
    temporal_concat = {}

    for key in in_dict:
        temporal_concat[key] = {}
        for seq in in_dict[key]:
                    temporal_concat[key][seq] = [concatenateTemporal(in_dict[key][seq])]
    return temporal_concat


def nodesMatch(n1, n_list):
    for i, node in enumerate(n_list, start=0):
        if node[1] == n1:
            return True, i

    return False, 0


def calculateTemporalEdges(full_gr, nodes, index_list, _relations, spatial_map):
    temporal_rel = _relations[spatial_map.index("temporal")]
    old_nodes = []
    node_cnt = index_list[-1]


    for index in index_list:
        old_nodes.append(full_gr.nodes[index])

    for index in index_list:
        match, ind = nodesMatch(full_gr.nodes[index], nodes)
        if match:
            full_gr.add_edge(index, node_cnt, edge_attr=temporal_rel)
            node_cnt += 1

def concatenateTemporal(graphs, _relations, spatial_map):
    graph_nx = nx.DiGraph()
    graph_nx.graph["features"] = graphs[0].graph['features']
    node_cnt = 0
    node_index_list = []

    for i, graph in enumerate(graphs, start=0):

        if len(node_index_list) > 0:
            calculateTemporalEdges(graph_nx, graph.nodes(data=True), node_index_list, _relations, spatial_map)

        node_index_list = []

        for node in graph.nodes(data=True):
            graph_nx.add_node(node_cnt, x=node[1]['x'])
            node_index_list.append(node_cnt)
            node_cnt += 1

        for edge in graph.edges():
            graph_nx.add_edge(node_index_list[edge[0]], node_index_list[edge[1]], edge_attr=graph.get_edge_data(edge[0], edge[1])['edge_attr'])

    empty_list = []
    for node in graph_nx.nodes(data=True):
        if node[1] == {}:
            empty_list.append(node[0])

    for node in empty_list:
        graph_nx.remove_node(node)


    return graph_nx
