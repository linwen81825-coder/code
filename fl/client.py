import copy

import torch
import torch.nn as nn

from .bayes_utils import cfg_get, count_bayes_evidence_entries, run_expert_sgld_fit


def _extract_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)



def _backup_torch_rng_state():
    rng_state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _restore_torch_rng_state(rng_state):
    torch.set_rng_state(rng_state["cpu"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _backup_requires_grad_state(model):
    return {name: param.requires_grad for name, param in model.named_parameters()}


def _restore_requires_grad_state(model, requires_grad_state):
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad_state.get(name, True))


def _log_bayes_evidence_timing(
    config,
    before_train_restore_done,
    train_final_mean_applied=False,
    missing_train_final_mean_keys=0,
):
    sgld_timing = str(cfg_get(config, "bayes_sgld_timing", "after_train")).lower()
    evidence_mean_source = str(cfg_get(config, "bayes_evidence_mean_source", "sgld_mean")).lower()
    evidence_precision_source = "pre_sgld" if sgld_timing == "before_train" else "post_train_sgld"
    print(
        "[BayesEvidenceTiming] "
        f"bayes_evidence_mean_source={evidence_mean_source} "
        f"evidence_timing={sgld_timing} "
        f"evidence_precision_source={evidence_precision_source} "
        f"train_final_mean_applied={str(bool(train_final_mean_applied)).lower()} "
        f"missing_train_final_mean_keys={int(missing_train_final_mean_keys)} "
        f"before_train_restore_done={str(bool(before_train_restore_done)).lower()}"
    )


def _resolve_bayes_cache_device(config, device):
    requested = str(cfg_get(config, "bayes_cache_device", "cuda")).lower()
    torch_device = torch.device(device)
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch_device.type == "cuda" and torch.cuda.is_available():
            return torch_device
        return torch.device("cpu")
    if requested == "auto":
        evidence_batches = int(cfg_get(config, "bayes_evidence_batches", 4))
        if torch_device.type == "cuda" and torch.cuda.is_available() and evidence_batches <= 4:
            return torch_device
        return torch.device("cpu")
    raise ValueError(f"Unsupported bayes_cache_device: {requested}")


def _move_cache_tensor(tensor, cache_device):
    if cache_device.type == "cuda":
        return tensor.detach().to(device=cache_device).clone()
    return tensor.detach().cpu().clone()


def _add_layer_stats(total_stats, layer_stats):
    for layer_id, stats in layer_stats.items():
        layer_key = str(layer_id)
        target = total_stats.setdefault(layer_key, {})
        for stat_key, value in stats.items():
            if stat_key == "sample_hits_by_expert":
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu()
                if stat_key not in target:
                    target[stat_key] = torch.zeros_like(value)
                target[stat_key] += value
            elif stat_key not in target:
                target[stat_key] = value


def _update_bayes_batch_cache(batch_cache_by_expert, inputs, labels, layer_stats, config, device):
    max_batches = max(int(cfg_get(config, "bayes_evidence_batches", 4)), 1)
    cache_device = _resolve_bayes_cache_device(config, device)
    cached_inputs = _move_cache_tensor(inputs, cache_device)
    cached_labels = _move_cache_tensor(labels, cache_device)

    for layer_id, stats in layer_stats.items():
        sample_hits_by_expert = stats.get("sample_hits_by_expert")
        if sample_hits_by_expert is None:
            continue

        sample_hits_by_expert = sample_hits_by_expert.detach().cpu()
        for expert_id in range(sample_hits_by_expert.size(1)):
            sample_hits = sample_hits_by_expert[:, expert_id]
            sample_indices = torch.nonzero(sample_hits > 0, as_tuple=False).flatten()
            if sample_indices.numel() == 0:
                continue

            expert_score = int(sample_hits[sample_indices].sum().item())
            device_indices = sample_indices.to(cache_device)
            expert_inputs = cached_inputs.index_select(0, device_indices).clone()
            expert_labels = cached_labels.index_select(0, device_indices).clone()
            expert_cache = (
                batch_cache_by_expert
                .setdefault(str(layer_id), {})
                .setdefault(str(expert_id), [])
            )
            expert_cache.append((expert_score, int(sample_indices.numel()), expert_inputs, expert_labels))
            expert_cache.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if len(expert_cache) > max_batches:
                del expert_cache[max_batches:]



def _collect_bayes_inputs(model, loader, device, config):
    layer_stats_total = {}
    batch_cache_by_expert = {}
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                model_output = model(inputs, return_stats=True)
                if not isinstance(model_output, dict):
                    continue
                batch_layer_stats = model_output.get("expert_stats_by_layer", {})
                _add_layer_stats(layer_stats_total, batch_layer_stats)
                _update_bayes_batch_cache(
                    batch_cache_by_expert=batch_cache_by_expert,
                    inputs=inputs,
                    labels=labels,
                    layer_stats=batch_layer_stats,
                    config=config,
                    device=device,
                )
    finally:
        model.train(was_training)
    return layer_stats_total, batch_cache_by_expert


def _apply_train_final_evidence_means(evidence_by_layer, train_final_state):
    train_final_mean_applied_count = 0
    missing_train_final_mean_keys = 0
    for expert_map in evidence_by_layer.values():
        if not isinstance(expert_map, dict):
            continue
        for expert_evidence in expert_map.values():
            if not isinstance(expert_evidence, dict):
                continue
            mean_state = expert_evidence.get("mean_state")
            precision_state = expert_evidence.get("precision_state")
            if not isinstance(mean_state, dict) or not isinstance(precision_state, dict):
                continue
            applied_count = 0
            missing_count = 0
            for key, sgld_mean in mean_state.items():
                if key not in precision_state:
                    continue
                train_final_mean = train_final_state.get(key)
                if (
                    not torch.is_tensor(train_final_mean)
                    or not torch.is_tensor(sgld_mean)
                    or not torch.is_tensor(precision_state[key])
                    or train_final_mean.shape != sgld_mean.shape
                    or train_final_mean.shape != precision_state[key].shape
                ):
                    missing_train_final_mean_keys += 1
                    missing_count += 1
                    continue
                mean_state[key] = train_final_mean.detach().cpu().clone()
                train_final_mean_applied_count += 1
                applied_count += 1
            if applied_count > 0:
                sgld_diag = expert_evidence.get("sgld_diag")
                if isinstance(sgld_diag, dict):
                    sgld_diag.update({
                        "mean_state_source": "train_final",
                        "precision_state_source": "pre_sgld",
                        "train_final_mean_applied": True,
                        "missing_train_final_mean_keys": missing_count,
                    })
    return (
        train_final_mean_applied_count,
        missing_train_final_mean_keys,
        train_final_mean_applied_count > 0,
    )


def _get_active_expert_refs(layer_stats, config):
    min_tokens = int(cfg_get(config, "bayes_min_expert_tokens", 128))
    active_experts = []
    for layer_id, stats in layer_stats.items():
        usage = stats.get("expert_activations")
        if usage is None:
            continue
        for expert_id, expert_usage in enumerate(usage.tolist()):
            expert_usage = int(expert_usage)
            if expert_usage >= min_tokens:
                active_experts.append((str(layer_id), str(expert_id), expert_usage))
    return active_experts


def _get_expert_batch_cache(batch_cache_by_expert, layer_id, expert_id):
    layer_cache = batch_cache_by_expert.get(str(layer_id), {})
    expert_cache = layer_cache.get(str(expert_id), [])
    batch_cache = []
    for cache_entry in expert_cache:
        _, _, cached_inputs, cached_labels = cache_entry
        batch_cache.append((cached_inputs, cached_labels))
    return batch_cache


def _get_expert_param_prefix(layer_id, expert_id):
    return f"{layer_id}.experts.{expert_id}."


def _backup_expert_params(model, layer_id, expert_id):
    prefix = _get_expert_param_prefix(layer_id, expert_id)
    backup = {}
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            backup[name] = param.detach().clone()
    return backup


def _restore_expert_params(model, backup):
    if not backup:
        return
    param_dict = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in backup.items():
            param_dict[name].copy_(value)


def _extract_bayesian_evidence(model, layer_stats, batch_cache_by_expert, criterion, device, config):
    active_experts = _get_active_expert_refs(layer_stats, config)
    cached_expert_count = sum(
        1
        for layer_cache in batch_cache_by_expert.values()
        for expert_cache in layer_cache.values()
        if len(expert_cache) > 0
    )
    print(
        "[BayesEvidence] "
        f"active_experts={len(active_experts)} cached_experts={cached_expert_count}"
    )
    if not active_experts:
        print("[BayesEvidence] skip: no expert reached bayes_min_expert_tokens")
        return {}

    evidence_by_layer = {}
    for layer_id, expert_id, usage in active_experts:
        batch_cache = _get_expert_batch_cache(batch_cache_by_expert, layer_id, expert_id)
        if len(batch_cache) == 0:
            print(
                "[BayesEvidence] "
                f"skip layer={layer_id} expert={expert_id}: empty cached batches"
            )
            continue

        expert_backup = _backup_expert_params(model, layer_id, expert_id)
        try:
            mean_state, precision_state, sgld_diag = run_expert_sgld_fit(
                model=model,
                batch_cache=batch_cache,
                criterion=criterion,
                layer_id=layer_id,
                expert_id=expert_id,
                device=device,
                steps=cfg_get(config, "bayes_sgld_steps", 10),
                burnin=cfg_get(config, "bayes_sgld_burnin", 5),
                alp=cfg_get(config, "bayes_sgld_lr", 0.00005),
                var_floor=cfg_get(config, "bayes_sgld_var_floor", 0.0),
                precision_eps=cfg_get(config, "bayes_precision_eps", 1.0e-12),
                precision_source=cfg_get(config, "bayes_precision_source", "sgld_variance"),
                sgld_fit_mode=cfg_get(config, "bayes_sgld_fit_mode", "adam_noise"),
                precision_mode=cfg_get(config, "bayes_precision_mode", "floor_inverse"),
                precision_min=cfg_get(config, "bayes_weighted_precision_min", 1.0e-8),
                precision_max=cfg_get(config, "bayes_weighted_precision_max", 1.0e8),
                movement_diag=cfg_get(config, "bayes_sgld_movement_diag", False),
                stuck_rel_threshold=cfg_get(config, "bayes_sgld_stuck_rel_threshold", 1.0e-5),
                stuck_abs_threshold=cfg_get(config, "bayes_sgld_stuck_abs_threshold", 1.0e-7),
            )
        except ValueError as exc:
            print(
                "[BayesEvidence] "
                f"skip layer={layer_id} expert={expert_id}: {exc}"
            )
            continue
        finally:
            _restore_expert_params(model, expert_backup)

        evidence_by_layer.setdefault(str(layer_id), {})[str(expert_id)] = {
            "usage": int(usage),
            "num_batches": len(batch_cache),
            "mean_state": mean_state,
            "precision_state": precision_state,
            "sgld_diag": sgld_diag,
        }
        print(
            "[BayesEvidence] "
            f"layer={layer_id} expert={expert_id} usage={usage} "
            f"precision_mean={sgld_diag.get('precision_mean')} "
            f"precision_min={sgld_diag.get('precision_min')} "
            f"precision_max={sgld_diag.get('precision_max')}"
        )

    evidence_count = count_bayes_evidence_entries(evidence_by_layer)
    if evidence_count == 0:
        print("[BayesEvidence] skip: collected evidence is empty")
    return evidence_by_layer


def local_train(
    global_model,
    loader,
    device,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    collect_bayes_evidence=False,
    bayes_config=None,
):
    bayes_config = bayes_config or {}
    bayes_sgld_timing = str(cfg_get(bayes_config, "bayes_sgld_timing", "after_train")).lower()
    bayes_evidence_mean_source = str(cfg_get(bayes_config, "bayes_evidence_mean_source", "sgld_mean")).lower()
    if collect_bayes_evidence and bayes_sgld_timing not in {"after_train", "before_train"}:
        raise ValueError("bayes_sgld_timing must be one of: after_train, before_train")
    if collect_bayes_evidence and bayes_evidence_mean_source not in {"sgld_mean", "train_final"}:
        raise ValueError("bayes_evidence_mean_source must be one of: sgld_mean, train_final")
    if (
        collect_bayes_evidence
        and bayes_sgld_timing == "after_train"
        and bayes_evidence_mean_source == "train_final"
    ):
        raise ValueError(
            "bayes_evidence_mean_source=train_final currently requires "
            "bayes_sgld_timing=before_train"
        )

    model = copy.deepcopy(global_model).to(device)
    model.train()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss, n_processed = 0.0, 0
    client_sample_count = len(loader.dataset)
    bayes_batch_cache_by_expert = {}
    bayes_layer_stats = {}
    before_train_bayes_evidence = None
    before_train_restore_done = False

    if collect_bayes_evidence and bayes_sgld_timing == "before_train":
        model_state_before_sgld = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        requires_grad_state = _backup_requires_grad_state(model)
        rng_state_before_sgld = _backup_torch_rng_state()
        try:
            before_train_layer_stats, before_train_batch_cache = _collect_bayes_inputs(
                model=model,
                loader=loader,
                device=device,
                config=bayes_config,
            )
            before_train_bayes_evidence = _extract_bayesian_evidence(
                model=model,
                layer_stats=before_train_layer_stats,
                batch_cache_by_expert=before_train_batch_cache,
                criterion=criterion,
                device=device,
                config=bayes_config,
            )
        finally:
            model.load_state_dict(model_state_before_sgld)
            _restore_requires_grad_state(model, requires_grad_state)
            _restore_torch_rng_state(rng_state_before_sgld)
            model.train()
            before_train_restore_done = True

    collect_train_bayes_stats = collect_bayes_evidence and bayes_sgld_timing == "after_train"
    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if collect_train_bayes_stats:
                model_output = model(x, return_stats=True)
            else:
                model_output = model(x)
            logits = _extract_logits(model_output)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * x.size(0)
            n_processed += x.size(0)

            if collect_train_bayes_stats and isinstance(model_output, dict):
                batch_layer_stats = model_output.get("expert_stats_by_layer", {})
                _add_layer_stats(bayes_layer_stats, batch_layer_stats)
                _update_bayes_batch_cache(
                    batch_cache_by_expert=bayes_batch_cache_by_expert,
                    inputs=x,
                    labels=y,
                    layer_stats=batch_layer_stats,
                    config=bayes_config,
                    device=device,
                )

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    avg_loss = total_loss / max(n_processed, 1)

    bayes_evidence = {}
    if collect_bayes_evidence:
        if before_train_bayes_evidence is None:
            bayes_evidence = _extract_bayesian_evidence(
                model=model,
                layer_stats=bayes_layer_stats,
                batch_cache_by_expert=bayes_batch_cache_by_expert,
                criterion=criterion,
                device=device,
                config=bayes_config,
            )
            _log_bayes_evidence_timing(
                bayes_config,
                before_train_restore_done=False,
            )
        else:
            bayes_evidence = before_train_bayes_evidence
            train_final_mean_applied = False
            missing_train_final_mean_keys = 0
            if bayes_evidence_mean_source == "train_final":
                _, missing_train_final_mean_keys, train_final_mean_applied = _apply_train_final_evidence_means(
                    evidence_by_layer=bayes_evidence,
                    train_final_state=state,
                )
            _log_bayes_evidence_timing(
                bayes_config,
                before_train_restore_done=before_train_restore_done,
                train_final_mean_applied=train_final_mean_applied,
                missing_train_final_mean_keys=missing_train_final_mean_keys,
            )
        if (
            torch.device(device).type == "cuda"
            and torch.cuda.is_available()
            and _as_bool(cfg_get(bayes_config, "bayes_empty_cache_after_client_evidence", False))
        ):
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state, client_sample_count, avg_loss, bayes_evidence
