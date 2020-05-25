import Utils.config as conf
import glob
import re
import os
import json
import networkx as nx

# Load BAC config
cfg = conf.get_bac_cfg()

def get_object_list(path):
    # ['DREHER', 'bimacs_derived_data', 'subject_1', 'task_5_k_cereals', 'take_3', 'spatial_relations', 'frame_0.json']
    final_path = path[0] + '/bimacs_derived_data_3D/' + path[2] + '/' + path[3] + '/' + path[4] + '/3d_objects/' + path[6]
    with open(final_path) as f:
        data = json.load(f)

    temp_dict = {}
    cnt = 0

    for obj in data:
        temp_dict[cnt] = obj['class_name']
        cnt += 1

    return temp_dict


# Takes a single JSON file and outputs a graph
def json_to_graph(input_path, target):
    graph = nx.DiGraph()

    node_list_path = os.path.normpath(input_path).split(os.sep)
    node_list = get_object_list(node_list_path)

    # Load JSON Object to list
    with open(input_path) as f:
          data = json.load(f)

    for index, name in node_list.items():
        graph.add_node(index, x=cfg._objects[cfg.objects.index(name)])

    # Populate the graph with nodes and edges
    for obj in data:
        relation_name = obj['relation_name']
        graph.add_edge(obj['object_index'], obj['subject_index'], edge_attr=cfg._relations[cfg.spatial_map.index(relation_name)])

    # If the ground truth contain null value set the action to undefined
    if(target is None):
        return -1
        #graph.graph['features'] = _actions[action_map.index('undefined')]
    else:
        #action_map.index(original_index[target])
        graph.graph["features"] = cfg._actions[cfg.action_map.index(original_index[target])]

    if len(graph.nodes()) == 0:
        print("------- NO NODES")
        return -1

    if len(graph.edges()) == 0:
        print("------- NO EDGES")
        return -1


    return graph


def get_target_action(cnt, ground_truth):
    # Find target by comparing the frame count with ground truth
    for index, item in enumerate(ground_truth['right_hand']):
        if(index % 2 == 0 and index != 0):
            if(cnt <= item):
                return ground_truth['right_hand'][index-1]
    return 'Not found'

def take_to_graph_list(path):
    graph_list = []
    # Get ground truth path
    gt_name = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')][0]
    gt_path = path + gt_name

    # Extract all json files in spatial_relations and sort them
    json_files = [pos_json for pos_json in os.listdir(path + 'spatial_relations') if pos_json.endswith('.json')]
    json_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Load ground truth
    with open(gt_path) as f:
          ground_truth = json.load(f)

    for file in json_files:
        frame_cnt = int(re.search(r'\d+', file).group())
        graphs = json_to_graph(path + 'spatial_relations/' + file, get_target_action(frame_cnt, ground_truth))

        if graphs != -1:
            graph_list.append(graphs)
        else:
            print("file:", file, "frame_cnt:", frame_cnt, "gt:", ground_truth)

    return graph_list


def generate_graphs(MAIN_PATH="./DREHER/bimacs_derived_data", _seperate=True):

    if _seperate:
        print("Creating graphs to dict.")
        all_data = {}
    else:
        print("Appending graphs to array.")
        all_data = []

    # Iterate subjects
    for dic in list(glob.glob(MAIN_PATH+'/*/')):
        # Iterate tasks
        for sub_dic in list(glob.glob(dic+'/*/')):
            # Iterate takes
            for sub_sub_dic in list(glob.glob(sub_dic+'/*/')):
                # Get subject number
                sub = int(re.search(r'\d+', dic).group())
                # Get task number
                task = int(re.search(r'\d+', sub_dic[len(dic):]).group())
                # Get take number
                take = int(re.search(r'\d+', sub_sub_dic[len(sub_dic):]).group())
                name = "take_" + str(sub) + "_" + str(task) + "_" + str(take)

                if _seperate:
                    all_data[name] = take_to_graph_list(sub_sub_dic)
                else:
                    all_data += take_to_graph_list(sub_sub_dic)

    return all_data


#test = generate_graphs()
