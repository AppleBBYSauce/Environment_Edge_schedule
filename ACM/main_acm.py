import gc
from torch.utils.data import DataLoader, Dataset

from model_acm import FLGnnH
import torch
from sklearn.metrics import f1_score
from loader_acm import acm
import numpy as np
from torch.cuda.amp import autocast as autocast, GradScaler
from tqdm import tqdm
from ogb.nodeproppred.evaluate import Evaluator
from itertools import count

device = "cuda"

evaluator = lambda y_true, y_pre: f1_score(y_true, y_pre, average="micro")


class GDataset(Dataset):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def __getitem__(self, item):
        return self.idx[item]

    def __len__(self):
        return self.idx.size(0)


def Grid_Search(config):
    res = []
    items = list(config.keys())

    def grid_search(select: dict, deep: int = 0):
        if deep == len(config):
            res.append(select)
        else:
            k = items[deep]
            for v in config[k]:
                tmp = select.copy()
                tmp.update({k: v})
                grid_search(tmp, deep + 1)

    grid_search(dict())
    return res


def fit(config: dict, dataset):
    mpath_feats, agg_msg, label, train_idx, val_idx, test_idx, total_idx = dataset
    m = FLGnnH(
        semantic_num=len(mpath_feats),
        hidden=config["hidden"],
        out_channels=max(label) + 1,
        num_mf=config["num_mf"], feature_projector_layers=config["feature_projector_layers"],
        val_interval=config["val_interval"], cross=config["cross"],
        window_size=config["window_size"], stride_size=config["stride_size"],
        dropout=config["dropout"], refine_ratio=config["refine_ratio"],
        choquet=config["choquet"], choquet_concat=config["choquet_concat"], choquet_heads=config["choquet_heads"],
        norm=config["norm"], concat=config["concat"], fix=config["fix"], fuzzy=config["fuzzy"],
        data_size=mpath_feats, residual=config["residual"], target=config["target"]
    ).to(device)

    epoch = config.get("epoch", 10)
    optim_config = config.get("optim", {"lr": 0.05, "weight_decay": 5e-4})
    critical = {
        "regress": torch.nn.MSELoss,
        "binary_classify": torch.nn.BCEWithLogitsLoss,
        "multi_classify": torch.nn.CrossEntropyLoss,
    }[config.get("type")]()

    train_loader = DataLoader(dataset=GDataset(idx=train_idx), batch_size=config.get("batch_size", 256), shuffle=True)
    val_loader = DataLoader(dataset=GDataset(idx=val_idx), batch_size=10000, shuffle=False)
    test_loader = DataLoader(dataset=GDataset(idx=test_idx), batch_size=10000, shuffle=False)

    optimizer = torch.optim.AdamW(**optim_config, params=m.parameters())
    reduce_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                 T_max=epoch // 10,
                                                                 eta_min=0.001)

    min_loss = None
    scaler = GradScaler()
    m.train()
    pbar = tqdm(total=epoch)
    best_val = 0
    best_test = 0
    best_epoch = 0
    best_test_ = 0
    best_e = 0
    for e in range(1, epoch + 1):
        loss_recorder = []
        for idx in train_loader:
            if idx.size(0) == 1:
                continue
            x = {i: j[idx].to(device) for i, j in agg_msg.items()}
            y = label[idx].to(device)
            optimizer.zero_grad()
            with autocast():
                pre_y = m(x)
                loss = critical(pre_y, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_recorder.append(loss.item())

        # m.ChI.standardize()
        reduce_schedule.step()
        gc.collect()
        torch.cuda.empty_cache()

        # == test phase ==

        m.eval()
        with torch.no_grad():
            critical = torch.nn.CrossEntropyLoss()

            test_loss_recorder = []
            val_loss_recorder = []

            # == valid ==
            true = []
            pre = []
            for idx in val_loader:
                x = {i: j[idx].to(device) for i, j in agg_msg.items()}
                y = label[idx].to(device)
                pre_y = m(x)
                pre.append(pre_y.detach().cpu())
                true.append(y.detach().cpu())
            pre = torch.cat(pre, dim=0).argmax(dim=1)
            true = torch.cat(true, dim=0)
            val_res = evaluator(true.unsqueeze(-1), pre.unsqueeze(-1))

            # == test ==
            true = []
            pre = []
            for idx in test_loader:
                x = {i: j[idx].to(device) for i, j in agg_msg.items()}
                y = label[idx].to(device)
                pre_y = m(x)
                pre.append(pre_y.detach().cpu())
                true.append(y.detach().cpu())
            pre = torch.cat(pre, dim=0).argmax(dim=1)
            true = torch.cat(true, dim=0)
            test_res = evaluator(true.unsqueeze(-1), pre.unsqueeze(-1))

            if val_res > best_val:
                best_val = round(val_res, 8)
                best_test = round(test_res, 8)
                best_epoch = e
            if test_res > best_test_:
                best_test_ = test_res
                best_e = e

        m.train()
        pbar.set_description(
            f'epoch: {e} / {epoch}, '
            f'train_loss_mean: {round(np.mean(loss_recorder) * config["batch_size"], 8)}, '
            f'test_acc: {test_res}, val_acc: {val_res}, best_t: {best_test_}, best_e: {best_e}, '
            f'best_epoch: {best_epoch}, best_val: {best_val}, best_test: {best_test}')
        pbar.update()
    pbar.close()
    gc.collect()
    torch.cuda.empty_cache()
    # m.ChI.draw_hasse_diagram()
    print(m.ChI.shapley_value())
    return best_test


if __name__ == '__main__':

    configure = {
        "in_channels": [128],
        "hidden": [90],
        "window_size": [3],
        "stride_size": [3],
        "concat": [False],
        "feature_projector_layers": [2],
        "residual": [False],
        "target": ["P"],
        "refine_ratio": [1],
        "dropout": [0.5],
        "cross": [1],
        "num_mf": [2],
        "fix": [True],
        "fuzzy": [False],
        "norm": [False],
        "val_interval": [[-1, 1]],
        "choquet": [True],
        "choquet_heads": [2],
        "choquet_concat": [False],
        "num_hop": [3],
        "optim": [
            {"lr": 0.005, "weight_decay": 5e-5},
        ],
        "batch_size": [32],
        "type": ["multi_classify"],
        "epoch": [20],
    }
    for cfg in Grid_Search(configure):
        res = []
        print(cfg)
        for exp in range(5):
            dataset = acm(num_hop=cfg["num_hop"], HGB_dataset=True, target_node=cfg["target"],
                          extra_path=["P", "PA", "PP", "PT", "PS", "PAP", "PTP", "PPT", "PSP"],
                          focus_extra_path=True)
            print(dataset[0].keys())
            sub_res = fit(cfg, dataset)
            res.append(sub_res)
        gc.collect()
        torch.cuda.empty_cache()
        print(np.mean(res), np.std(res))
        gc.collect()
        torch.cuda.empty_cache()
