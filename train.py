import argparse
import csv
import gc
import json
import os
import sys
import traceback
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import torch
import yaml
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


class _LogOnlyStream:
    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, data):
        if not data:
            return 0
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self):
        self.log_file.flush()


class ConsoleLogRouter:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _LogOnlyStream(self.log_file)
        sys.stderr = _LogOnlyStream(self.log_file)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.log_file.flush()
        self.log_file.close()
        return False

    def console_summary(self, message):
        line = f"{message}\n"
        self.log_file.write(line)
        self.log_file.flush()
        self._stdout.write(line)
        self._stdout.flush()

    def log_exception(self):
        traceback.print_exc(file=self.log_file)
        self.log_file.flush()
        traceback.print_exc(file=self._stderr)
        self._stderr.flush()


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


def resolve_run_dir(cfg):
    run_name = cfg.get("run_name")
    if run_name:
        root = str(cfg.get("output_root", cfg.get("output_dir", "outputs")))
        return os.path.join(root, str(run_name))
    return str(cfg.get("output_dir", "outputs"))


def save_config_used(cfg):
    path = os.path.join(cfg["output_dir"], "config_used.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _xlsx_col_name(index):
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(row_idx, col_idx, value):
    ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_minimal_xlsx(records, path):
    headers = list(records[0].keys()) if records else []
    rows = []
    if headers:
        rows.append(headers)
        rows.extend([[record.get(header) for header in headers] for record in records])

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(row_idx, col_idx, value) for col_idx, value in enumerate(row))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    sheet_data = "".join(sheet_rows)

    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="results" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        "xl/styles.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>""",
        "xl/worksheets/sheet1.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{sheet_data}</sheetData></worksheet>""",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        for name, content in files.items():
            xlsx.writestr(name, content)


def write_results_outputs(records, cfg):
    results_path = os.path.join(cfg["output_dir"], "results.xlsx")
    legacy_path = os.path.join(
        cfg["output_dir"],
        f"MoEFed_results_{cfg['dataset']}_clients{cfg['num_clients']}_"
        f"experts{cfg['num_experts']}_nonexpert-{cfg['non_expert_agg_method']}_"
        f"expert-{cfg['expert_agg_method']}.xlsx",
    )
    xlsx_paths = [results_path]
    if legacy_path != results_path:
        xlsx_paths.append(legacy_path)

    try:
        import pandas as pd

        for path in xlsx_paths:
            pd.DataFrame(records).to_excel(path, index=False)
    except (ModuleNotFoundError, ImportError):
        for path in xlsx_paths:
            write_minimal_xlsx(records, path)

    csv_path = os.path.join(cfg["output_dir"], "results.csv")
    fieldnames = list(records[0].keys()) if records else []
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)

    return results_path, legacy_path


def run_training(args, cfg, router):
    set_seed(cfg["seed"])

    print("Config:")
    print(cfg)

    device = resolve_device(cfg.get("device", "auto"))
    cfg["resolved_device"] = str(device)
    save_config_used(cfg)
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
    rounds_path = os.path.join(cfg["output_dir"], "rounds.jsonl")
    clients_path = os.path.join(cfg["output_dir"], "clients.jsonl")
    open(rounds_path, "w", encoding="utf-8").close()
    open(clients_path, "w", encoding="utf-8").close()

    print(f"{'Round':>5} | {'LR':>8} | {'AvgLoss':>9} | {'TestAcc':>8} | {'BestAcc':>8}")
    print("-" * 52)

    for rnd in range(start_round, cfg["rounds"] + 1):
        current_lr = cfg["lr"]
        chosen = np.random.choice(cfg["num_clients"], m, replace=False).tolist()
        print(f"[RoundClients] round={rnd} clients={chosen}")
        for cid in chosen:
            append_jsonl(
                clients_path,
                {
                    "round": rnd,
                    "client_id": int(cid),
                    "sample_count": int(len(client_indices[cid])),
                },
            )

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

        round_record = {
            "Round": rnd,
            "LR": current_lr,
            "AvgLoss": avg_loss,
            "TestAcc": acc,
            "BestAcc": best_acc,
            "BayesUpdatedExperts": aggregation_metrics.get("updated_experts"),
            "BayesSkippedExperts": aggregation_metrics.get("skipped_experts"),
        }
        records.append(round_record)
        append_jsonl(rounds_path, round_record)
        print(
            f"{rnd:5d} | {current_lr:8.4f} | {avg_loss:9.4f} | "
            f"{acc:8.2f} | {best_acc:8.2f}"
        )
        print(
            "[Metrics] "
            f"round={rnd} acc={acc:.4f} best_acc={best_acc:.4f} "
            f"avg_loss={avg_loss:.4f}"
        )
        router.console_summary(
            f"round={rnd} acc={acc:.4f} best_acc={best_acc:.4f} avg_loss={avg_loss:.4f}"
        )

        del new_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_path, legacy_path = write_results_outputs(records, cfg)
    print(f"Done. Best Acc: {best_acc:.2f}%")
    print(f"[Export] saved to: {results_path}")
    if legacy_path != results_path:
        print(f"[Export] legacy saved to: {legacy_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["output_dir"] = resolve_run_dir(cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_path = os.path.join(cfg["output_dir"], "train.log")
    with ConsoleLogRouter(log_path) as router:
        try:
            run_training(args, cfg, router)
        except BaseException:
            router.log_exception()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
