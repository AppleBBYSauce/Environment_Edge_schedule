import gc
import torch
from torch_geometric.datasets import HGBDataset
import torch_geometric.transforms as T
from torch_scatter import scatter_mean
from collections import defaultdict
import numpy as np
import os

device = "cpu"





def abbreviate(s, r, d):
    src_type = s[0].upper()
    dst_type = d[0].upper()
    rel_type = f"{src_type}{dst_type}"
    return src_type, rel_type, dst_type


def acm(num_hop=2, extra_path=None, embed_size=128, name="", target_node="A", focus_extra_path=False,
         HGB_dataset=True):
    transform = T.Compose([T.NormalizeFeatures()])
    # transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.NormalizeFeatures()])
    g = HGBDataset(name="ACM", root=r"./dataset", pre_transform=transform)[0]
    g["term"].x = torch.randn(size=(g["term"].num_nodes, embed_size))
    g["term"].num_features = embed_size


    g["abb_edge_types"] = []
    g["abb_node_types"] = set()

    # create abbreviation alias
    for edge_type in g.edge_types:
        src_type, rel_type, dst_type = edge_type
        abb_src_type, abb_rel_type, abb_dst_type = abbreviate(src_type, rel_type, dst_type)
        g[abb_rel_type] = g[edge_type].edge_index
        if "rev" not in rel_type:
            g[abb_src_type] = g[src_type]
            g[abb_dst_type] = g[dst_type]

        g["abb_edge_types"].append(abb_rel_type)
        g["abb_node_types"].add(abb_src_type)
        g["abb_node_types"].add(abb_dst_type)
    mpath_feats, agg_msg = pre_propagate(g, num_hop=num_hop, extra_meta_path=extra_path, name=name,
                                         target_node=target_node, focus_extra_path=focus_extra_path)
    val_ratio = 0.2
    total_idx = torch.arange(start=0, end=g[target_node].num_nodes)
    train_val_idx = total_idx[g[target_node]["train_mask"]]
    shuffle_index = np.arange(start=0, stop=train_val_idx.size(0), step=1)
    np.random.shuffle(shuffle_index)
    train_val_idx = train_val_idx[shuffle_index]
    split = int(train_val_idx.shape[0] * val_ratio)
    train_idx = train_val_idx[split:]
    val_idx = train_val_idx[:split]
    test_idx = total_idx[g[target_node]["test_mask"]]
    return mpath_feats, agg_msg, g[target_node].y, train_idx, val_idx, test_idx, total_idx


def search_possible_meta_path(target_node, max_hop, edge_rel):
    res = set()
    adj = defaultdict(list)
    for i in edge_rel:
        adj[i[0]].append(i[1])

    def searcher(cur_path="", hop=0):
        if len(cur_path) > 1 and cur_path[-1] == target_node:
            res.add(cur_path)
        if hop == max_hop:
            return
        elif len(cur_path) > 0:
            for nx in adj[cur_path[-1]]:
                searcher(cur_path + nx, hop + 1)
        else:
            for nx in adj.keys():
                searcher(nx, 0)

    searcher()
    return res


def pre_propagate(g, target_node="A", num_hop=1, extra_meta_path=None, name="agg_msg", focus_extra_path=False):
    extra_meta_path = {"", } if not extra_meta_path else {i[::-1] for i in extra_meta_path}
    edge_rel = g["abb_edge_types"]
    if not focus_extra_path:
        target_meta_path = search_possible_meta_path(edge_rel=edge_rel, target_node=target_node, max_hop=num_hop)
        target_meta_path.update(set(extra_meta_path) if extra_meta_path is not None else set())
    else:
        target_meta_path = set(extra_meta_path)
    target_meta_path.add(target_node)
    aggregation_res = {i: g[i].x for i in g.abb_node_types}
    max_hop = max(max(len(i) for i in extra_meta_path), num_hop)
    for length in range(2, max_hop + 1):
        cur_path = [path[:length] for path in target_meta_path if len(path) >= length]
        for path in cur_path:
            if path in aggregation_res or len(path) < length:
                continue
            src_type, edge_type, tar_type = path[:-1], path[-2] + path[-1], path[-1]
            edge = g[edge_type]
            src = aggregation_res[src_type][edge[0]]
            aggregation_res[path] = scatter_mean(src=src, index=edge[1], dim=0, dim_size=g[tar_type].num_nodes)

    # clear intermediary path
    aggregation_res = {i[::-1]: j for i, j in aggregation_res.items() if i in target_meta_path}
    gc.collect()
    torch.cuda.empty_cache()
    mpath_feats = {}
    for type, msg in aggregation_res.items():
        mpath_feats[type] = msg.size(1)
    aggregation_res = [mpath_feats, aggregation_res]
    # torch.save(aggregation_res, r"./dataset_HGB/dblp/" + name)
    return aggregation_res


if __name__ == '__main__':
    acm(target_node='P', num_hop=3, name="agg.pt", HGB_dataset=True)

