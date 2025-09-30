import torch
from collections import defaultdict, OrderedDict
import numba
import numpy as np
import networkx as nx
import torch
from typing import List, Tuple
import dgl


def _group_by(keys, values) -> dict:
    """Group values by keys.

    :param keys: list of keys
    :param values: list of values
    A key value pair i is defined by (key_list[i], value_list[i]).
    :return: OrderedDict where key value pairs have been grouped by key.

     """
    result = defaultdict(list)
    for key, value in zip(keys.tolist(), values.tolist()):
        result[tuple(key)].append(value)
    for key, value in result.items():
        result[key] = torch.IntTensor(sorted(value))
    return OrderedDict(result)


def index_KvsAll(dataset: "Dataset", split: str, key: str):
    """Return an index for the triples in split (''train'', ''valid'', ''test'')
    from the specified key (''sp'' or ''po'' or ''so'') to the indexes of the
    remaining constituent (''o'' or ''s'' or ''p'' , respectively.)

    The index maps from `tuple' to `torch.LongTensor`.

    The index is cached in the provided dataset under name `{split}_sp_to_o` or
    `{split}_po_to_s`, or `{split}_so_to_p`. If this index is already present, does not
    recompute it.

    """
    value = None
    if key == "sp":
        key_cols = [0, 1]
        value_column = 2
        value = "o"
    elif key == "po":
        key_cols = [1, 2]
        value_column = 0
        value = "s"
    elif key == "so":
        key_cols = [0, 2]
        value_column = 1
        value = "p"
    else:
        raise ValueError()

    name = split + "_" + key + "_to_" + value
    if not dataset._indexes.get(name):
        triples = dataset.split(split)
        dataset._indexes[name] = _group_by(
            triples[:, key_cols], triples[:, value_column]
        )

    dataset.config.log(
        "{} distinct {} pairs in {}".format(len(dataset._indexes[name]), key, split),
        prefix="  ",
    )

    return dataset._indexes.get(name)


def index_KvsAll_to_torch(index):
    """Convert `index_KvsAll` indexes to pytorch tensors.

    Returns an nx2 keys tensor (rows = keys), an offset vector
    (row = starting offset in values for corresponding key),
    a values vector (entries correspond to values of original
    index)

    Afterwards, it holds:
        index[keys[i]] = values[offsets[i]:offsets[i+1]]
    """
    keys = torch.tensor(list(index.keys()), dtype=torch.int)
    values = torch.cat(list(index.values()))
    offsets = torch.cumsum(
        torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
    )
    return keys, values, offsets


def index_frequency_percent(dataset):
    name = "fre"
    if not dataset._indexes.get(name):
        fre = []
        entities_fre = {}
        train_triples = dataset.split('train')
        for i in range(dataset.num_entities()):
            entities_fre[i] = 0
        for tri in train_triples:
            s, p, t, o = tri.tolist()
            if s in entities_fre:
                entities_fre[s] += 1
        for i in range(dataset.num_entities()):
            fre.append(entities_fre[i] / dataset.num_entities())
        dataset._indexes[name] = fre
    dataset.config.log("Entities_fre index finished", prefix="  ")
    return dataset._indexes[name]


def index_neighbor_multidig(dataset):
    name = "neighbor"
    if not dataset._indexes.get(name):
        train_triples = dataset.split('train')
        G = nx.MultiDiGraph()
        for tri in train_triples:
            s, p, o, t= tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)

        # import powerlaw # Power laws are probability distributions with the form:p(x)∝x−α
        max_neighbor_num = 1000
        all_neighbor = torch.zeros((dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long)
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)

        # get the information about all edges
        edges_attributes = G.edges(data=True)
        neighbor_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_types_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_times_all = [[] for _ in range(dataset.num_entities())]

        rng = np.random.default_rng()

        for s, o, data in edges_attributes:
            neighbor_all[s].append(o)
            neighbor_all[o].append(s)
            neighbor_edge_types_all[s].append(data["type"] + dataset.num_relations())
            neighbor_edge_times_all[s].append(data["time"])
            neighbor_edge_types_all[o].append(data["type"])
            neighbor_edge_times_all[o].append(data["time"])

        for s in range(dataset.num_entities()):
            if s not in G: continue
            rand_permut = rng.permutation(len(neighbor_all[s]))
            neighbor = np.asarray(neighbor_all[s])[rand_permut]
            neighbor_edge_types = np.asarray(neighbor_edge_types_all[s])[rand_permut]
            neighbor_edge_times = np.asarray(neighbor_edge_times_all[s])[rand_permut]
            neighbor = neighbor[:max_neighbor_num]
            neighbor_edge_types = neighbor_edge_types[:max_neighbor_num]
            neighbor_edge_times = neighbor_edge_times[:max_neighbor_num]
            all_neighbor[s, 0, 0:len(neighbor)] = torch.tensor(neighbor, dtype=torch.long)
            all_neighbor[s, 1, 0:len(neighbor)] = torch.tensor(neighbor_edge_types, dtype=torch.long)
            all_neighbor[s, 2, 0:len(neighbor)] = torch.tensor(neighbor_edge_times, dtype=torch.long)
            all_neighbor_num[s] = len(neighbor)
        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Neighbors index finished", prefix="  ")

    return dataset._indexes.get(name)


def index_temporal_paths(dataset):
    """Build an index for temporal paths in the dataset, facilitating message passing on the history temporal graph.

    The index stores, for each entity, its neighbors with corresponding edge types and timestamps from the training set.
    This allows efficient filtering of edges within a historical time window [t-m+1, t] for a query at time t+1.

    Args:
        dataset: The dataset object containing the training triples and metadata.

    Returns:
        tuple: (all_neighbor, all_neighbor_num)
            - all_neighbor: Tensor of shape (num_entities, 3, max_neighbor_num), where for each entity s:
                - [s, 0, :]: neighbor entities
                - [s, 1, :]: edge types (p or p + num_relations for inverse)
                - [s, 2, :]: edge timestamps
            - all_neighbor_num: Tensor of shape (num_entities,), number of neighbors per entity.
    """
    name = "temporal_paths"
    if not dataset._indexes.get(name):
        # Load training triples
        train_triples = dataset.split('train')

        # Build a multi-directed graph with temporal edges
        G = nx.MultiDiGraph()
        for tri in train_triples:
            s, p, o, t = tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)

        # Set maximum number of neighbors per entity (adjustable)
        max_neighbor_num = 1000
        all_neighbor = torch.zeros((dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long)
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)

        # Initialize lists to collect neighbor information
        edges_attributes = G.edges(data=True)
        neighbor_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_types_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_times_all = [[] for _ in range(dataset.num_entities())]

        # Random number generator for shuffling neighbors
        rng = np.random.default_rng()

        # Populate neighbor lists with outgoing and incoming edges
        for s, o, data in edges_attributes:
            # Outgoing edge: s -> o with type p, stored as o being s's neighbor with inverse type
            neighbor_all[s].append(o)
            neighbor_edge_types_all[s].append(data["type"] + dataset.num_relations())  # Inverse relation
            neighbor_edge_times_all[s].append(data["time"])
            # Incoming edge: o -> s with type p, stored as s being o's neighbor with original type
            neighbor_all[o].append(s)
            neighbor_edge_types_all[o].append(data["type"])
            neighbor_edge_times_all[o].append(data["time"])

        # Process each entity to populate the index
        for s in range(dataset.num_entities()):
            if s not in G:
                continue
            # Randomly shuffle neighbors to avoid bias in truncation
            rand_permut = rng.permutation(len(neighbor_all[s]))
            neighbor = np.asarray(neighbor_all[s])[rand_permut]
            neighbor_edge_types = np.asarray(neighbor_edge_types_all[s])[rand_permut]
            neighbor_edge_times = np.asarray(neighbor_edge_times_all[s])[rand_permut]

            # Truncate to max_neighbor_num
            neighbor = neighbor[:max_neighbor_num]
            neighbor_edge_types = neighbor_edge_types[:max_neighbor_num]
            neighbor_edge_times = neighbor_edge_times[:max_neighbor_num]

            # Store in tensor
            num_neighbors = len(neighbor)
            all_neighbor[s, 0, :num_neighbors] = torch.tensor(neighbor, dtype=torch.long)
            all_neighbor[s, 1, :num_neighbors] = torch.tensor(neighbor_edge_types, dtype=torch.long)
            all_neighbor[s, 2, :num_neighbors] = torch.tensor(neighbor_edge_times, dtype=torch.long)
            all_neighbor_num[s] = num_neighbors

        # Cache the index
        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Temporal paths index finished", prefix="  ")
    return dataset._indexes.get(name)


def index_neighbor_dig(dataset):
    name = "neighbor"
    if not dataset._indexes.get(name):
        train_triples = dataset.split('train')
        G = nx.DiGraph()
        for tri in train_triples:
            s, p, o, t= tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)
        max_neighbor_num = 300
        all_neighbor = torch.zeros((dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long)
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)
        rng = np.random.default_rng()
        for s in range(dataset.num_entities()):
            if s not in G:
                continue
            suc = list(G.successors(s))
            pre = list(G.predecessors(s))
            suc_edge_types = [G.get_edge_data(s, v)['type'] + dataset.num_relations() for v in suc]
            pre_edge_types = [G.get_edge_data(v, s)['type'] for v in pre]
            suc_edge_times = [G.get_edge_data(s, v)['time'] for v in suc]
            pre_edge_times = [G.get_edge_data(v, s)['time'] for v in pre]
            rand_permut = rng.permutation(len(suc) + len(pre))
            neighbor = np.asarray(suc + pre)[rand_permut]
            neighbor_edge_types = np.asarray(suc_edge_types + pre_edge_types)[rand_permut]
            neighbor_edge_times = np.asarray(suc_edge_times + pre_edge_times)[rand_permut]
            neighbor = neighbor[:max_neighbor_num]
            neighbor_edge_types = neighbor_edge_types[:max_neighbor_num]
            neighbor_edge_times = neighbor_edge_times[:max_neighbor_num]
            all_neighbor[s, 0, 0:len(neighbor)] = torch.tensor(neighbor, dtype=torch.long)
            all_neighbor[s, 1, 0:len(neighbor)] = torch.tensor(neighbor_edge_types, dtype=torch.long)
            all_neighbor[s, 2, 0:len(neighbor)] = torch.tensor(neighbor_edge_times, dtype=torch.long)
            all_neighbor_num[s] = len(neighbor)
        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Neighbors index finished", prefix="  ")

    return dataset._indexes.get(name)


def index_relation_types(dataset):
    """Classify relations into 1-N, M-1, 1-1, M-N.

    According to Bordes et al. "Translating embeddings for modeling multi-relational
    data.", NIPS13.

    Adds index `relation_types` with list that maps relation index to ("1-N", "M-1",
    "1-1", "M-N").

    """
    if "relation_types" not in dataset._indexes:
        # 2nd dim: num_s, num_distinct_po, num_o, num_distinct_so, is_M, is_N
        relation_stats = torch.zeros((dataset.num_relations(), 6))
        for index, p in [
            (dataset.index("train_sp_to_o"), 1),
            (dataset.index("train_po_to_s"), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0 + p * 2] = relation_stats[
                    prefix[p], 0 + p * 2
                ] + len(labels)
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
        relation_types = []
        for i in range(dataset.num_relations()):
            relation_types.append(
                "{}-{}".format(
                    "1" if relation_stats[i, 4].item() == 0 else "M",
                    "1" if relation_stats[i, 5].item() == 0 else "N",
                )
            )

        dataset._indexes["relation_types"] = relation_types

    return dataset._indexes["relation_types"]


def index_relations_per_type(dataset):
    if "relations_per_type" not in dataset._indexes:
        relations_per_type = {}
        for i, k in enumerate(dataset.index("relation_types")):
            relations_per_type.setdefault(k, set()).add(i)
        dataset._indexes["relations_per_type"] = relations_per_type
    else:
        relations_per_type = dataset._indexes["relations_per_type"]

    dataset.config.log("Loaded relation index")
    for k, relations in relations_per_type.items():
        dataset.config.log(
            "{} relations of type {}".format(len(relations), k), prefix="  "
        )

    return relations_per_type


def index_frequency_percentiles(dataset, recompute=False):
    """
    :return: dictionary mapping from
    {
        'subject':
        {25%, 50%, 75%, top} -> set of entities
        'relations':
        {25%, 50%, 75%, top} -> set of relations
        'object':
        {25%, 50%, 75%, top} -> set of entities
    }
    """
    if "frequency_percentiles" in dataset._indexes and not recompute:
        return
    subject_stats = torch.zeros((dataset.num_entities(), 1))
    relation_stats = torch.zeros((dataset.num_relations(), 1))
    object_stats = torch.zeros((dataset.num_entities(), 1))
    for (s, p, o) in dataset.split("train"):
        subject_stats[s] += 1
        relation_stats[p] += 1
        object_stats[o] += 1
    result = dict()
    for arg, stats, num in [
        (
            "subject",
            [
                i
                for i, j in list(
                    sorted(enumerate(subject_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
        (
            "relation",
            [
                i
                for i, j in list(
                    sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_relations(),
        ),
        (
            "object",
            [
                i
                for i, j in list(
                    sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
    ]:
        for percentile, (begin, end) in [
            ("25%", (0.0, 0.25)),
            ("50%", (0.25, 0.5)),
            ("75%", (0.5, 0.75)),
            ("top", (0.75, 1.0)),
        ]:
            if arg not in result:
                result[arg] = dict()
            result[arg][percentile] = set(stats[int(begin * num) : int(end * num)])
    dataset._indexes["frequency_percentiles"] = result


class IndexWrapper:
    """Wraps a call to an index function so that it can be pickled"""

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.kwargs = kwargs

    def __call__(self, dataset: "Dataset", **kwargs):
        self.fun(dataset, **self.kwargs)


def _invert_ids(dataset, obj: str):
    if not f"{obj}_id_to_index" in dataset._indexes:
        ids = dataset.load_map(f"{obj}_ids")
        inv = {v: k for k, v in enumerate(ids)}
        dataset._indexes[f"{obj}_id_to_index"] = inv
    else:
        inv = dataset._indexes[f"{obj}_id_to_index"]
    dataset.config.log(f"Indexed {len(inv)} {obj} ids", prefix="  ")


def index_all_edges(dataset):
    name = "all_edges"
    if not dataset._indexes.get(name):
        all_neighbor, all_neighbor_num = dataset.index("neighbor")
        all_sources = []
        all_targets = []
        all_relations = []
        all_times = []
        for s in range(dataset.num_entities()):
            num_neighbors = all_neighbor_num[s].item()
            if num_neighbors > 0:
                sources = [s] * num_neighbors
                targets = all_neighbor[s, 0, :num_neighbors].tolist()
                relations = all_neighbor[s, 1, :num_neighbors].tolist()
                times = all_neighbor[s, 2, :num_neighbors].tolist()
                all_sources.extend(sources)
                all_targets.extend(targets)
                all_relations.extend(relations)
                all_times.extend(times)
        all_sources = torch.tensor(all_sources, dtype=torch.long)
        all_targets = torch.tensor(all_targets, dtype=torch.long)
        all_relations = torch.tensor(all_relations, dtype=torch.long)
        all_times = torch.tensor(all_times, dtype=torch.long)
        dataset._indexes[name] = (all_sources, all_targets, all_relations, all_times)
    return dataset._indexes.get(name)


def index_query_subgraphs(dataset, k_hop=2, max_subgraph_size=500):
    """Build subgraphs for each entity in the dataset for query-specific processing.
    
    For each entity, this function constructs a k-hop subgraph containing neighboring 
    entities, relations, and temporal information. This supports localized graph neural 
    network processing for temporal knowledge graph completion.
    
    Args:
        dataset: The dataset object containing the training triples and metadata.
        k_hop: Number of hops to include in each subgraph (default: 2)
        max_subgraph_size: Maximum number of nodes in each subgraph to control memory usage
        
    Returns:
        dict: A dictionary mapping each entity ID to its subgraph information:
            {
                entity_id: {
                    'nodes': list of entity IDs in the subgraph,
                    'edges': list of (src, rel, dst, time) tuples,
                    'node_mapping': mapping from original entity ID to subgraph node ID,
                    'center_node': subgraph node ID of the query entity
                }
            }
    """
    name = "query_subgraphs"
    if not dataset._indexes.get(name):
        train_triples = dataset.split('train')
        
        # Build global graph for efficient neighbor queries
        G = nx.MultiDiGraph()
        for tri in train_triples:
            s, p, o, t = tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, relation=p, time=t)
            G.add_edge(o, s, relation=p + dataset.num_relations(), time=t)  # Add inverse edge
        
        subgraph_dict = {}
        
        for entity in range(dataset.num_entities()):
            if entity not in G:
                # Handle isolated entities
                subgraph_dict[entity] = {
                    'nodes': [entity],
                    'edges': [],
                    'node_mapping': {entity: 0},
                    'center_node': 0
                }
                continue
                
            # BFS to collect k-hop neighbors
            visited = set()
            queue = [(entity, 0)]  # (node, hop_distance)
            subgraph_nodes = set([entity])
            
            while queue and len(subgraph_nodes) < max_subgraph_size:
                current_node, hop = queue.pop(0)
                
                if current_node in visited or hop >= k_hop:
                    continue
                    
                visited.add(current_node)
                
                # Add neighbors
                for neighbor in G.neighbors(current_node):
                    if neighbor not in subgraph_nodes and len(subgraph_nodes) < max_subgraph_size:
                        subgraph_nodes.add(neighbor)
                        queue.append((neighbor, hop + 1))
            
            # Create node mapping
            nodes_list = list(subgraph_nodes)
            node_mapping = {node: idx for idx, node in enumerate(nodes_list)}
            
            # Extract edges within the subgraph
            edges = []
            for src in subgraph_nodes:
                for dst in G.neighbors(src):
                    if dst in subgraph_nodes:
                        # Get all edges between src and dst (handling multi-edges)
                        edge_data = G.get_edge_data(src, dst)
                        if edge_data:
                            for edge_key, edge_attr in edge_data.items():
                                edges.append((
                                    node_mapping[src],
                                    edge_attr['relation'],
                                    node_mapping[dst], 
                                    edge_attr['time']
                                ))
            
            subgraph_dict[entity] = {
                'nodes': nodes_list,
                'edges': edges,
                'node_mapping': node_mapping,
                'center_node': node_mapping[entity]
            }
        
        dataset._indexes[name] = subgraph_dict
        
    dataset.config.log("Query subgraphs index finished", prefix="  ")
    return dataset._indexes.get(name)


def index_temporal_subgraphs(dataset, k_hop=2, time_window=5, max_subgraph_size=500):
    """Build temporal-aware subgraphs for each query.
    
    This function creates subgraphs that are filtered by temporal constraints,
    ensuring that only edges within a specified time window are included.
    
    Args:
        dataset: The dataset object
        k_hop: Number of hops in the subgraph
        time_window: Maximum time difference to include edges
        max_subgraph_size: Maximum nodes in subgraph
        
    Returns:
        dict: Temporal subgraph information for each entity
    """
    name = "temporal_subgraphs"
    if not dataset._indexes.get(name):
        train_triples = dataset.split('train')
        
        # Build temporal adjacency lists
        temporal_neighbors = defaultdict(lambda: defaultdict(list))
        
        for tri in train_triples:
            s, p, o, t = tri.tolist()
            temporal_neighbors[s][t].append((o, p))
            temporal_neighbors[o][t].append((s, p + dataset.num_relations()))
        
        subgraph_dict = {}
        
        for entity in range(dataset.num_entities()):
            entity_subgraphs = {}
            
            # Get all timestamps where this entity appears
            entity_times = list(temporal_neighbors[entity].keys())
            
            for query_time in entity_times:
                # Define time window
                start_time = max(0, query_time - time_window)
                end_time = query_time
                
                # Collect nodes and edges within time window
                subgraph_nodes = set([entity])
                edges = []
                
                # BFS with temporal constraints
                queue = [(entity, 0)]
                visited = set()
                
                while queue and len(subgraph_nodes) < max_subgraph_size:
                    current_node, hop = queue.pop(0)
                    
                    if current_node in visited or hop >= k_hop:
                        continue
                        
                    visited.add(current_node)
                    
                    # Look for neighbors in the time window
                    for t in range(start_time, end_time + 1):
                        if t in temporal_neighbors[current_node]:
                            for neighbor, relation in temporal_neighbors[current_node][t]:
                                if len(subgraph_nodes) < max_subgraph_size:
                                    subgraph_nodes.add(neighbor)
                                    if hop + 1 < k_hop:
                                        queue.append((neighbor, hop + 1))
                
                # Create node mapping and edge list
                nodes_list = list(subgraph_nodes)
                node_mapping = {node: idx for idx, node in enumerate(nodes_list)}
                
                # Extract edges
                temporal_edges = []
                for node in subgraph_nodes:
                    for t in range(start_time, end_time + 1):
                        if t in temporal_neighbors[node]:
                            for neighbor, relation in temporal_neighbors[node][t]:
                                if neighbor in subgraph_nodes:
                                    temporal_edges.append((
                                        node_mapping[node],
                                        relation,
                                        node_mapping[neighbor],
                                        t
                                    ))
                
                entity_subgraphs[query_time] = {
                    'nodes': nodes_list,
                    'edges': temporal_edges,
                    'node_mapping': node_mapping,
                    'center_node': node_mapping[entity],
                    'query_time': query_time
                }
            
            subgraph_dict[entity] = entity_subgraphs
        
        dataset._indexes[name] = subgraph_dict
        
    dataset.config.log("Temporal subgraphs index finished", prefix="  ")
    return dataset._indexes.get(name)


def build_dgl_subgraph(subgraph_info, num_relations, device='cpu'):
    """Convert subgraph information to DGL graph format.
    
    Args:
        subgraph_info: Subgraph information from index_query_subgraphs
        num_relations: Total number of relations in the dataset
        device: Device to place the graph on
        
    Returns:
        dgl.DGLGraph: The subgraph in DGL format with node and edge features
    """
    nodes = subgraph_info['nodes']
    edges = subgraph_info['edges']
    
    if len(edges) == 0:
        # Handle single node case
        g = dgl.graph(([], []), num_nodes=len(nodes))
        g.ndata['entity_id'] = torch.tensor(nodes, dtype=torch.long)
        g.ndata['center_mask'] = torch.zeros(len(nodes), dtype=torch.bool)
        g.ndata['center_mask'][subgraph_info['center_node']] = True
        return g.to(device)
    
    # Extract edge information
    src_nodes, relations, dst_nodes, times = zip(*edges)
    
    # Create DGL graph
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(nodes))
    
    # Add node features
    g.ndata['entity_id'] = torch.tensor(nodes, dtype=torch.long)
    g.ndata['center_mask'] = torch.zeros(len(nodes), dtype=torch.bool)
    g.ndata['center_mask'][subgraph_info['center_node']] = True
    
    # Add edge features
    g.edata['relation'] = torch.tensor(relations, dtype=torch.long)
    g.edata['time'] = torch.tensor(times, dtype=torch.long)
    
    return g.to(device)


def create_default_index_functions(dataset: "Dataset"):
    for split in dataset.files_of_type("triples"):
        for key, value in [("sp", "o"), ("po", "s"), ("so", "p")]:
            # self assignment needed to capture the loop var
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_KvsAll, split=split, key=key
            )
    dataset.index_functions["neighbor"] = index_neighbor_multidig
    dataset.index_functions["temporal_paths"] = index_temporal_paths
    dataset.index_functions["query_subgraphs"] = index_query_subgraphs
    dataset.index_functions["temporal_subgraphs"] = index_temporal_subgraphs
    dataset.index_functions["relation_types"] = index_relation_types
    dataset.index_functions["relations_per_type"] = index_relations_per_type
    dataset.index_functions["frequency_percentiles"] = index_frequency_percentiles
    dataset.index_functions["fre"] = index_frequency_percent
    dataset.index_functions["all_edges"] = index_all_edges


    for obj in ["entity", "relation"]:
        dataset.index_functions[f"{obj}_id_to_index"] = IndexWrapper(
            _invert_ids, obj=obj
        )


@numba.njit
def where_in(x, y, not_in=False):
    """Retrieve the indices of the elements in x which are also in y.

    x and y are assumed to be 1 dimensional arrays.

    :params: not_in: if True, returns the indices of the of the elements in x
    which are not in y.

    """
    # np.isin is not supported in numba. Also: "i in y" raises an error in numba
    # setting njit(parallel=True) slows down the function
    list_y = set(y)
    return np.where(np.array([i in list_y for i in x]) != not_in)[0]
