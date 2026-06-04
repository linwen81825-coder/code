import argparse
import gc
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import DATASET_CFG, get_dataset, partition_dirichlet
from fl import (
    build_initial_bayes_state,
    load_resume_checkpoint,
    run_fl_round,
    save_bayes_state,
    save_resume_checkpoint,
    save_server_model,
    summarize_param_groups,
    uses_expert_bayes_meta,
)
from model import MoEFedModel
from utils import evaluate, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print("Config:")
    print(cfg)

    device = resolve_device(cfg.get("device", "auto"))
    cfg["resolved_device"] = str(device)
    print(f"Device: {device}")

    train_ds, test_ds = get_dataset(cfg["dataset"], cfg["data_root"])
    client_indices = partition_dirichlet(
        train_ds, cfg["num_clients"], cfg["beta"], cfg["seed"]
    )
    client_loaders = [
        DataLoader(
            Subset(train_ds, idx),
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
        for idx in client_indices
    ]
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["test_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"train dataset size: {len(train_ds)}")
    print(f"test dataset size: {len(test_ds)}")
    print(f"number of clients: {len(client_indices)}")
    print(f"each client sample count: {[len(idx) for idx in client_indices]}")
    print(f"number of client_loaders: {len(client_loaders)}")
    print(f"test_loader batch size: {test_loader.batch_size}")

    dataset_cfg = DATASET_CFG[cfg["dataset"]]
    global_model = MoEFedModel(
        in_channels=dataset_cfg["in_channels"],
        num_classes=dataset_cfg["num_classes"],
        img_size=dataset_cfg["img_size"],
        num_experts=cfg["num_experts"],
        topk=cfg["topk"],
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters())
    print(f"[Model] Total params: {n_params:,}")
    summarize_param_groups(global_model.state_dict())

    bayes_enabled = uses_expert_bayes_meta(cfg)
    bayes_state = None
    start_round = 1
    print(
        "[Bayes] "
        f"expert_bayes_meta_enabled={bayes_enabled} "
        f"agg_method={cfg['agg_method']} "
        f"expert_agg_method={cfg['expert_agg_method']}"
    )

    if cfg.get("resume", False):
        completed_round, bayes_state = load_resume_checkpoint(global_model, cfg)
        start_round = completed_round + 1
        print(f"[Resume] loaded completed_round={completed_round}")
    elif bayes_enabled:
        bayes_state = build_initial_bayes_state(global_model, cfg)
        save_bayes_state(bayes_state, cfg)
        print("[Bayes] initialized server_bayes_state.pth")

    records = []
    best_acc = 0.0
    m = max(1, int(cfg["num_clients"] * cfg["frac"]))

    print(f"{'Round':>5} | {'LR':>8} | {'AvgLoss':>9} | {'TestAcc':>8} | {'BestAcc':>8}")
    print("-" * 52)

    for rnd in range(start_round, cfg["rounds"] + 1):
        current_lr = cfg["lr"]
        chosen = np.random.choice(cfg["num_clients"], m, replace=False).tolist()

        new_state, avg_loss, updated_bayes_state, aggregation_metrics = run_fl_round(
            global_model=global_model,
            client_loaders=client_loaders,
            chosen_clients=chosen,
            device=device,
            local_epochs=cfg["local_epochs"],
            lr=current_lr,
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
            non_expert_agg_method=cfg["non_expert_agg_method"],
            expert_agg_method=cfg["expert_agg_method"],
            agg_method=cfg["agg_method"],
            bayes_state=bayes_state,
            bayes_config=cfg,
        )
        global_model.load_state_dict(new_state)
        save_server_model(global_model, cfg)
        if updated_bayes_state is not None:
            bayes_state = updated_bayes_state
            save_bayes_state(bayes_state, cfg)
        save_resume_checkpoint(global_model, bayes_state, cfg, rnd)

        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

        records.append(
            {
                "Round": rnd,
                "LR": current_lr,
                "AvgLoss": avg_loss,
                "TestAcc": acc,
                "BestAcc": best_acc,
                "BayesUpdatedExperts": aggregation_metrics.get("updated_experts"),
                "BayesSkippedExperts": aggregation_metrics.get("skipped_experts"),
            }
        )
        print(
            f"{rnd:5d} | {current_lr:8.4f} | {avg_loss:9.4f} | "
            f"{acc:8.2f} | {best_acc:8.2f}"
        )
        print(
            "[Metrics] "
            f"round={rnd} acc={acc:.2f} best_acc={best_acc:.2f} "
            f"avg_loss={avg_loss:.4f}"
        )

        del new_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(
        cfg["output_dir"],
        f"MoEFed_results_{cfg['dataset']}_clients{cfg['num_clients']}_"
        f"experts{cfg['num_experts']}_nonexpert-{cfg['non_expert_agg_method']}_"
        f"expert-{cfg['expert_agg_method']}.xlsx",
    )
    try:
        import pandas as pd

        pd.DataFrame(records).to_excel(
            out_path,
            index=False,
        )
    except ModuleNotFoundError:
        out_path = out_path.rsplit(".", 1)[0] + ".csv"
        fieldnames = list(records[0].keys()) if records else []
        with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"Done. Best Acc: {best_acc:.2f}%")
    print(f"[Export] saved to: {out_path}")


if __name__ == "__main__":
    main()
