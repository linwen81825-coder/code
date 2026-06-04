import os

import numpy as np
import torch

from .aggregators import aggregate_split_model
from .bayes_utils import (
    build_initial_bayes_state,
    cfg_get,
    count_bayes_evidence_entries,
    uses_expert_bayes_meta,
)
from .client import local_train


def get_model_dir(config):
    return os.path.join(str(cfg_get(config, "output_dir", "outputs")), "model")


def get_bayes_state_path(config):
    return os.path.join(get_model_dir(config), "server_bayes_state.pth")


def get_server_model_path(config):
    return os.path.join(get_model_dir(config), "server.pth")


def save_server_model(global_model, config):
    os.makedirs(get_model_dir(config), exist_ok=True)
    torch.save(
        {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        },
        get_server_model_path(config),
    )


def get_resume_checkpoint_path(config):
    configured = cfg_get(config, "resume_checkpoint_path", None)
    if configured in {None, "", "null", "None"}:
        return os.path.join(get_model_dir(config), "resume_checkpoint.pth")
    return str(configured)


def save_bayes_state(bayes_state, config):
    if bayes_state is None:
        return
    os.makedirs(get_model_dir(config), exist_ok=True)
    torch.save(bayes_state, get_bayes_state_path(config))


def load_bayes_state(config):
    path = get_bayes_state_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing server_bayes_state.pth: {path}")
    return torch.load(path, map_location="cpu")


def save_resume_checkpoint(global_model, bayes_state, config, completed_round):
    os.makedirs(get_model_dir(config), exist_ok=True)
    path = get_resume_checkpoint_path(config)
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "completed_round": int(completed_round),
        "server_model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        },
        "bayes_state": bayes_state,
        "agg_method": cfg_get(config, "agg_method", "decoupled_moe"),
        "expert_agg_method": cfg_get(config, "expert_agg_method", "uniform"),
    }
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_resume_checkpoint(global_model, config):
    path = get_resume_checkpoint_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume=True but checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    server_state = checkpoint.get("server_model_state_dict")
    if server_state is None:
        raise KeyError(f"Missing server_model_state_dict in resume checkpoint: {path}")
    global_model.load_state_dict(server_state)
    bayes_state = checkpoint.get("bayes_state")
    if uses_expert_bayes_meta(config) and bayes_state is None:
        raise RuntimeError("resume=True with expert_bayes_meta requires bayes_state in resume_checkpoint.pth")
    completed_round = int(checkpoint.get("completed_round", 0))
    return completed_round, bayes_state


def unpack_aggregation_output(aggregation_output):
    if isinstance(aggregation_output, dict) and "model_state" in aggregation_output:
        return (
            aggregation_output["model_state"],
            aggregation_output.get("bayes_state"),
            aggregation_output.get("metrics", {}),
        )
    return aggregation_output, None, {}


def _log_bayes_round(enabled, client_bayes_evidences, aggregation_metrics):
    print(f"[Bayes] expert_bayes_meta_enabled={enabled}")
    if not enabled:
        return

    evidence_counts = [
        count_bayes_evidence_entries(evidence)
        for evidence in (client_bayes_evidences or [])
    ]
    total_evidence = sum(evidence_counts)
    print(
        "[Bayes] "
        f"collected_evidence_total={total_evidence} "
        f"per_client={evidence_counts}"
    )
    if total_evidence == 0:
        print("[Bayes] skip: evidence is empty for this round")

    if not aggregation_metrics:
        return
    print(
        "[Bayes] "
        f"updated_experts={aggregation_metrics.get('updated_experts', 0)} "
        f"skipped_experts={aggregation_metrics.get('skipped_experts', 0)}"
    )
    expert_stats = aggregation_metrics.get("expert_meta_stats", {})
    for expert_ref, stats in expert_stats.items():
        calibration_mode = stats.get("precision_calibration_mode")
        calibration_summary = (
            ""
            if calibration_mode is None
            else f" precision_calibration_mode={calibration_mode}"
        )
        print(
            "[BayesExpert] "
            f"expert={expert_ref} status={stats.get('status')} "
            f"clients={stats.get('clients')} "
            f"precision_mean={stats.get('precision_mean')} "
            f"precision_min={stats.get('precision_min')} "
            f"precision_max={stats.get('precision_max')}"
            f"{calibration_summary}"
        )


def run_fl_round(
    global_model,
    client_loaders,
    chosen_clients,
    device,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    non_expert_agg_method,
    expert_agg_method,
    agg_method="decoupled_moe",
    bayes_state=None,
    bayes_config=None,
):
    config = bayes_config or {}
    bayes_enabled = uses_expert_bayes_meta({
        **config,
        "agg_method": agg_method,
        "expert_agg_method": expert_agg_method,
    })
    client_states = []
    client_samples = []
    client_losses = []
    client_bayes_evidences = []

    for cid in chosen_clients:
        state, sample_count, loss, bayes_evidence = local_train(
            global_model=global_model,
            loader=client_loaders[cid],
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            collect_bayes_evidence=bayes_enabled,
            bayes_config=config,
        )
        client_states.append(state)
        client_samples.append(sample_count)
        client_losses.append(loss)
        if bayes_enabled:
            client_bayes_evidences.append(bayes_evidence)

    aggregation_output = aggregate_split_model(
        global_model=global_model,
        client_states=client_states,
        client_samples=client_samples,
        non_expert_agg_method=non_expert_agg_method,
        expert_agg_method=expert_agg_method,
        agg_method=agg_method,
        client_bayes_evidence=client_bayes_evidences if bayes_enabled else None,
        bayes_state=bayes_state,
        bayes_config=config,
    )
    new_state, updated_bayes_state, aggregation_metrics = unpack_aggregation_output(aggregation_output)
    _log_bayes_round(bayes_enabled, client_bayes_evidences, aggregation_metrics)
    avg_loss = float(np.mean(client_losses))

    return new_state, avg_loss, updated_bayes_state, aggregation_metrics
