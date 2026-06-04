import collections
import copy
import math
import time

import torch

from .bayes_utils import (
    cfg_get,
    compute_optimal_local_posterior,
    compute_quadratic_meta_terms,
    get_bayes_expert_state,
    get_client_expert_evidence,
    group_expert_keys,
)
from .param_groups import split_state_keys


def normalize_key_aggregation_method(method):
    method = str(method or "uniform").lower()
    aliases = {
        "equal_avg": "uniform",
        "client_avg": "uniform",
        "sample_weighted_avg": "sample_weighted",
        "fedavg": "sample_weighted",
        "expert_fedavg": "sample_weighted",
    }
    method = aliases.get(method, method)
    if method not in {"uniform", "sample_weighted"}:
        raise ValueError(f"Unsupported aggregation method: {method}")
    return method


def aggregate_keys_uniform(global_state, client_states, keys):
    if not client_states:
        raise ValueError("client_states must not be empty")

    num_clients = len(client_states)
    aggregated = {}

    for key in keys:
        target = global_state[key]
        if torch.is_floating_point(target):
            avg = torch.zeros_like(target, device="cpu", dtype=torch.float32)
            for client_state in client_states:
                avg += client_state[key].detach().cpu().float() / num_clients
            aggregated[key] = avg.to(device=target.device, dtype=target.dtype)
        else:
            aggregated[key] = client_states[0][key].to(
                device=target.device,
                dtype=target.dtype,
            )

    return aggregated


def aggregate_keys_sample_weighted(global_state, client_states, client_samples, keys):
    if not client_samples:
        raise ValueError("client_samples must not be empty")
    if len(client_samples) != len(client_states):
        raise ValueError("client_samples length must match client_states length")

    total_samples = sum(client_samples)
    if total_samples <= 0:
        raise ValueError("sum(client_samples) must be positive")

    weights = [sample / total_samples for sample in client_samples]
    aggregated = {}

    for key in keys:
        target = global_state[key]
        if torch.is_floating_point(target):
            avg = torch.zeros_like(target, device="cpu", dtype=torch.float32)
            for client_state, weight in zip(client_states, weights):
                avg += client_state[key].detach().cpu().float() * weight
            aggregated[key] = avg.to(device=target.device, dtype=target.dtype)
        else:
            aggregated[key] = client_states[0][key].to(
                device=target.device,
                dtype=target.dtype,
            )

    return aggregated


def build_key_aggregator(method):
    method = normalize_key_aggregation_method(method)
    if method == "uniform":
        return aggregate_keys_uniform
    if method == "sample_weighted":
        return aggregate_keys_sample_weighted
    raise ValueError(f"Unsupported aggregation method: {method}")


def _aggregate_keys_by_method(global_state, client_states, client_samples, keys, method):
    method = normalize_key_aggregation_method(method)
    aggregator = build_key_aggregator(method)
    if method == "sample_weighted":
        return aggregator(global_state, client_states, client_samples, keys)
    return aggregator(global_state, client_states, keys)


def _summarize_tensors(tensors):
    values = []
    for value in tensors:
        if torch.is_tensor(value) and torch.is_floating_point(value):
            flat = value.detach().cpu().float().reshape(-1)
            flat = flat[torch.isfinite(flat)]
            if flat.numel() > 0:
                values.append(flat)
    if not values:
        return {"mean": None, "min": None, "max": None}
    vector = torch.cat(values)
    return {
        "mean": round(float(vector.mean().item()), 6),
        "min": round(float(vector.min().item()), 6),
        "max": round(float(vector.max().item()), 6),
    }



def _summarize_finite_scalars(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _resolve_precision_calibration_mode(config):
    mode = cfg_get(config, "bayes_precision_calibration_mode", None)
    if mode is None:
        mode = cfg_get(config, "bayes_weighted_precision_calibration", "none")
    return str(mode).lower()


def _sanitize_precision_tensor(value, config, replace_nonfinite=False):
    if not torch.is_tensor(value):
        return None
    value = value.detach()
    precision_min = float(cfg_get(config, "bayes_weighted_precision_min", 1.0e-8))
    precision_max = float(cfg_get(config, "bayes_weighted_precision_max", 1.0e8))
    if replace_nonfinite:
        value = torch.nan_to_num(
            value,
            nan=precision_min,
            posinf=precision_max,
            neginf=precision_min,
        )
    elif not torch.isfinite(value).all().item():
        return None
    return value.clamp(min=precision_min, max=precision_max)


def _sanitize_var_tensor(value, config):
    if not torch.is_tensor(value):
        return None
    value = value.detach()
    if not torch.isfinite(value).all().item():
        return None
    var_min = float(cfg_get(config, "bayes_weighted_var_min", 1.0e-8))
    var_max = float(cfg_get(config, "bayes_weighted_var_max", 1.0e8))
    return value.clamp(min=var_min, max=var_max)


def _build_shared_precision_calibration(client_payloads, expert_keys, config):
    mode = _resolve_precision_calibration_mode(config)
    target = float(cfg_get(config, "bayes_weighted_precision_target", 100.0))
    eps = max(float(cfg_get(config, "bayes_weighted_eps", 1.0e-8)), 1.0e-12)
    scales = collections.OrderedDict((key, 1.0) for key in expert_keys)
    raw_medians = []
    invalid_count = 0
    if mode == "median_target":
        for key in expert_keys:
            client_medians = []
            for payload in client_payloads:
                raw_precision = payload.get("precision_state", {}).get(key)
                if not torch.is_tensor(raw_precision):
                    continue
                raw_precision = raw_precision.detach()
                valid_values = raw_precision[torch.isfinite(raw_precision) & (raw_precision > 0)]
                if valid_values.numel() > 0:
                    client_medians.append(float(torch.quantile(valid_values.double(), 0.5).item()))
            if client_medians:
                raw_median = float(torch.quantile(torch.tensor(client_medians, dtype=torch.float64), 0.5).item())
            else:
                raw_median = None
            if raw_median is None or not math.isfinite(raw_median) or raw_median <= 0:
                invalid_count += 1
                continue
            scale = target / (raw_median + eps)
            if not math.isfinite(scale) or scale <= 0:
                invalid_count += 1
                continue
            scales[key] = float(scale)
            raw_medians.append(raw_median)

    scale_stats = _summarize_finite_scalars(scales.values())
    raw_median_stats = _summarize_finite_scalars(raw_medians)
    return scales, {
        "precision_calibration_mode": mode,
        "precision_target": target,
        "precision_scale_mean": scale_stats["mean"],
        "precision_scale_min": scale_stats["min"],
        "precision_scale_max": scale_stats["max"],
        "raw_precision_median_mean": raw_median_stats["mean"],
        "raw_precision_median_min": raw_median_stats["min"],
        "raw_precision_median_max": raw_median_stats["max"],
        "invalid_precision_calibration_count": invalid_count,
    }


def _apply_shared_precision_calibration(
    local_precision_state,
    expert_keys,
    scales,
    config,
    precision_clip_diag=None,
    replace_nonfinite=False,
):
    if not isinstance(local_precision_state, dict):
        return local_precision_state
    precision_min = float(cfg_get(config, "bayes_weighted_precision_min", 1.0e-8))
    precision_max = float(cfg_get(config, "bayes_weighted_precision_max", 1.0e8))
    effective_state = collections.OrderedDict()
    for key in expert_keys:
        raw_precision = local_precision_state.get(key)
        if not torch.is_tensor(raw_precision):
            effective_state[key] = raw_precision
            continue
        scaled_precision = raw_precision.detach().double() * float(scales.get(key, 1.0))
        if precision_clip_diag is not None:
            valid = torch.isfinite(scaled_precision)
            total_count = scaled_precision.numel() if replace_nonfinite else int(valid.sum().item())
            precision_clip_diag["precision_total_count"] += int(total_count)
            precision_clip_diag["precision_clip_max_count"] += int(
                ((scaled_precision > precision_max) & valid).sum().item()
            )
            precision_clip_diag["precision_clip_min_count"] += int(
                ((scaled_precision < precision_min) & valid).sum().item()
            )
        if replace_nonfinite:
            scaled_precision = torch.nan_to_num(
                scaled_precision,
                nan=precision_min,
                posinf=precision_max,
                neginf=precision_min,
            )
        elif not torch.isfinite(scaled_precision).all().item():
            effective_state[key] = raw_precision
            continue
        effective_state[key] = scaled_precision.clamp(min=precision_min, max=precision_max)
    return effective_state


def _compute_local_posterior_for_client(
    local_mean_state,
    local_precision_state,
    prior_mean_state,
    prior_var_state,
    n0,
    config,
):
    states = [local_mean_state, local_precision_state, prior_mean_state, prior_var_state]
    if not all(isinstance(state, dict) for state in states):
        return None

    m_star_state = collections.OrderedDict()
    v_star_state = collections.OrderedDict()
    sanitized_precision_state = collections.OrderedDict()
    with torch.no_grad():
        for key, prior_mean in prior_mean_state.items():
            local_mean = local_mean_state.get(key)
            local_precision = local_precision_state.get(key)
            prior_var = prior_var_state.get(key)
            tensors = [local_mean, local_precision, prior_mean, prior_var]
            if not all(torch.is_tensor(value) for value in tensors):
                return None
            if any(value.shape != prior_mean.shape for value in tensors):
                return None

            prior_mean = prior_mean.detach()
            local_mean = local_mean.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = local_precision.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            prior_var = prior_var.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = _sanitize_precision_tensor(local_precision, config)
            prior_var = _sanitize_var_tensor(prior_var, config)
            n0_tensor = torch.as_tensor(n0, device=prior_mean.device, dtype=prior_mean.dtype).detach()
            if (
                local_precision is None
                or prior_var is None
                or not torch.isfinite(local_mean).all().item()
                or not torch.isfinite(prior_mean).all().item()
                or not torch.isfinite(n0_tensor).all().item()
                or n0_tensor.numel() != 1
                or n0_tensor.item() < 0
            ):
                return None

            prior_eff_prec = n0_tensor / prior_var
            v_star = 1.0 / (local_precision + prior_eff_prec)
            m_star = v_star * (local_precision * local_mean + prior_eff_prec * prior_mean)
            if not torch.isfinite(v_star).all().item() or not torch.isfinite(m_star).all().item():
                return None
            m_star_state[key] = m_star
            v_star_state[key] = v_star
            sanitized_precision_state[key] = local_precision
    return m_star_state, v_star_state, sanitized_precision_state


def _compute_evidence_score_scalar(
    local_mean_state,
    local_precision_state,
    prior_mean_state,
    prior_var_state,
    m_star_state,
    v_star_state,
    n0,
    config,
):
    states = [
        local_mean_state,
        local_precision_state,
        prior_mean_state,
        prior_var_state,
        m_star_state,
        v_star_state,
    ]
    if not all(isinstance(state, dict) for state in states):
        return None

    f_terms = []
    g_terms = []
    eps = float(cfg_get(config, "bayes_weighted_eps", 1.0e-8))
    with torch.no_grad():
        for key, prior_mean in prior_mean_state.items():
            local_mean = local_mean_state.get(key)
            local_precision = local_precision_state.get(key)
            prior_var = prior_var_state.get(key)
            m_star = m_star_state.get(key)
            v_star = v_star_state.get(key)
            tensors = [local_mean, local_precision, prior_mean, prior_var, m_star, v_star]
            if not all(torch.is_tensor(value) for value in tensors):
                return None
            if any(value.shape != prior_mean.shape for value in tensors):
                return None

            prior_mean = prior_mean.detach()
            local_mean = local_mean.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = local_precision.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            prior_var = prior_var.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            m_star = m_star.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            v_star = v_star.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = _sanitize_precision_tensor(local_precision, config)
            prior_var = _sanitize_var_tensor(prior_var, config)
            v_star = _sanitize_var_tensor(v_star, config)
            n0_tensor = torch.as_tensor(n0, device=prior_mean.device, dtype=prior_mean.dtype).detach()
            if (
                local_precision is None
                or prior_var is None
                or v_star is None
                or not torch.isfinite(local_mean).all().item()
                or not torch.isfinite(prior_mean).all().item()
                or not torch.isfinite(m_star).all().item()
                or not torch.isfinite(n0_tensor).all().item()
                or n0_tensor.numel() != 1
                or n0_tensor.item() < 0
            ):
                return None

            f_terms.append((local_precision * v_star + local_precision * (m_star - local_mean).square()).mean())
            g_terms.append(
                (
                    torch.log(prior_var + eps)
                    - torch.log(v_star + eps)
                    + n0_tensor * v_star / (prior_var + eps)
                    + n0_tensor * (m_star - prior_mean).square() / (prior_var + eps)
                ).mean()
            )
        if len(f_terms) == 0:
            return None
        score = 0.5 * torch.stack(f_terms).mean() + 0.5 * torch.stack(g_terms).mean()
        if not torch.isfinite(score).item():
            return None
        return float(score.item())


def _scores_to_beta(scores, config):
    min_valid_clients = int(cfg_get(config, "bayes_weighted_min_valid_clients", 2))
    if len(scores) < min_valid_clients:
        return None

    score_values = []
    for score in scores:
        if torch.is_tensor(score):
            if score.numel() != 1:
                return None
            score = score.detach().cpu().item()
        try:
            score_values.append(float(score))
        except (TypeError, ValueError):
            return None
    score_tensor = torch.tensor(score_values, dtype=torch.float64)
    if not torch.isfinite(score_tensor).all().item():
        return None

    eps = float(cfg_get(config, "bayes_weighted_eps", 1.0e-8))
    sigma = score_tensor.std(unbiased=False)
    if sigma.item() < eps:
        return [1.0 / len(scores)] * len(scores)
    z = (score_tensor - score_tensor.mean()) / (sigma + eps)
    score_clip = float(cfg_get(config, "bayes_weighted_score_clip", 3.0))
    score_tau = float(cfg_get(config, "bayes_weighted_score_tau", 0.5))
    beta = torch.softmax(-score_tau * z.clamp(min=-score_clip, max=score_clip), dim=0)
    return beta.tolist()


class ExpertBayesMetaAggregator:
    def __init__(self, config):
        self.config = config or {}
        self.min_precision = 1.0e-6
        self.max_precision = 1.0e6
        self.max_n0 = 1.0e6
        self.gamma0_init = max(float(cfg_get(config, "bayes_gamma0_init", 1.0)), self.min_precision)
        self.n0_init = max(float(cfg_get(config, "bayes_n0_init", 1.0)), self.min_precision)
        self.meta_steps = max(int(cfg_get(config, "bayes_meta_steps", 2)), 1)
        self.meta_lr = float(cfg_get(config, "bayes_meta_lr", 0.0005))
        self.meta_update_mode = str(cfg_get(config, "bayes_meta_update_mode", "optimizer")).lower()
        self.weighted_score_tau = float(cfg_get(config, "bayes_weighted_score_tau", 0.5))
        self.weighted_score_clip = float(cfg_get(config, "bayes_weighted_score_clip", 3.0))
        self.weighted_var_rho = float(cfg_get(config, "bayes_weighted_var_rho", 0.05))
        self.weighted_eps = float(cfg_get(config, "bayes_weighted_eps", 1.0e-8))
        self.weighted_min_valid_clients = int(cfg_get(config, "bayes_weighted_min_valid_clients", 2))
        self.weighted_precision_min = float(cfg_get(config, "bayes_weighted_precision_min", 1.0e-8))
        self.weighted_precision_max = float(cfg_get(config, "bayes_weighted_precision_max", 1.0e8))
        self.weighted_precision_calibration = _resolve_precision_calibration_mode(config)
        self.weighted_precision_target = float(cfg_get(config, "bayes_weighted_precision_target", 100.0))
        self.v0_update_mode = str(cfg_get(config, "bayes_v0_update_mode", "precision_ema")).lower()
        self.weighted_var_min = float(cfg_get(config, "bayes_weighted_var_min", 1.0e-8))
        self.weighted_var_max = float(cfg_get(config, "bayes_weighted_var_max", 1.0e8))
        self.weighted_diag = bool(cfg_get(config, "bayes_weighted_diag", False))
        self.update_precision = bool(cfg_get(config, "bayes_update_precision", True))
        self.update_strength = bool(cfg_get(config, "bayes_update_strength", True))
        self.empty_cache_after_aggregation = bool(
            cfg_get(config, "bayes_empty_cache_after_aggregation", False)
        )
        self.meta_device = self._resolve_meta_device()

        precision_source = str(cfg_get(config, "bayes_precision_source", "sgld_variance")).lower()
        sgld_fit_mode = str(cfg_get(config, "bayes_sgld_fit_mode", "adam_noise")).lower()
        precision_mode = str(cfg_get(config, "bayes_precision_mode", "floor_inverse")).lower()
        if precision_source != "sgld_variance":
            raise ValueError("bayes_precision_source now only supports: sgld_variance")
        if sgld_fit_mode not in {"adam_noise", "sgd_noise"}:
            raise ValueError("bayes_sgld_fit_mode must be one of: adam_noise, sgd_noise")
        if precision_mode != "floor_inverse":
            raise ValueError("bayes_precision_mode now only supports: floor_inverse")
        if self.meta_update_mode not in {"optimizer", "closed_form_weighted"}:
            raise ValueError(
                "bayes_meta_update_mode must be one of: optimizer, closed_form_weighted"
            )
        if self.weighted_precision_calibration not in {
            "none",
            "median_target",
        }:
            raise ValueError(
                "bayes_precision_calibration_mode must be one of: "
                "none, median_target"
            )
        if self.v0_update_mode not in {"precision_ema", "fixed"}:
            raise ValueError("bayes_v0_update_mode must be one of: precision_ema, fixed")

        print(
            "[ExpertBayesMetaAggregator] "
            "bayes_precision_source=sgld_variance "
            f"bayes_sgld_noise_mode={sgld_fit_mode} "
            "bayes_precision_method=floor_inverse "
            f"bayes_meta_update_mode={self.meta_update_mode} "
            f"bayes_meta_device={self.meta_device} "
            f"bayes_precision_calibration_mode={self.weighted_precision_calibration} "
            f"bayes_v0_update_mode={self.v0_update_mode}"
        )

    def _resolve_meta_device(self):
        requested = str(cfg_get(self.config, "bayes_meta_device", "auto")).lower()
        base_device = str(cfg_get(self.config, "resolved_device", cfg_get(self.config, "device", "cpu")))
        if requested == "auto":
            if base_device.startswith("cuda") and torch.cuda.is_available():
                return torch.device(base_device)
            return torch.device("cpu")
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                print(
                    "[ExpertBayesMetaAggregator] CUDA requested for "
                    "bayes_meta_device but not available; fallback to CPU"
                )
                return torch.device("cpu")
            return torch.device(requested)
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported bayes_meta_device: {requested}")

    def aggregate(self, client_updates, global_state, expert_evidence, bayes_state):
        aggregate_start_time = time.perf_counter()
        if len(client_updates) == 0:
            raise ValueError("ExpertBayesMeta requires at least one client update")
        if expert_evidence is None:
            raise ValueError("ExpertBayesMeta requires expert_evidence from clients")
        if len(expert_evidence) != len(client_updates):
            raise ValueError("expert_evidence and client_updates must have the same length")
        if bayes_state is None:
            raise ValueError("ExpertBayesMeta requires bayes_state from server")

        updated_bayes_state = copy.deepcopy(bayes_state)
        expert_groups = group_expert_keys(global_state)
        expert_state = {}
        metrics = {
            "updated_experts": 0,
            "skipped_experts": 0,
            "evidence_clients": 0,
            "expert_param_groups": len(expert_groups),
            "local_posteriors": 0,
            "bayes_meta_device": str(self.meta_device),
            "expert_meta_stats": {},
        }

        for (layer_id, expert_id), expert_keys in expert_groups.items():
            expert_params, contributing_clients, local_posterior_count, expert_metric = (
                self._aggregate_expert_group(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    expert_keys=expert_keys,
                    global_state=global_state,
                    updated_bayes_state=updated_bayes_state,
                    expert_evidence=expert_evidence,
                )
            )
            metrics["expert_meta_stats"][f"{layer_id}.{expert_id}"] = expert_metric
            if contributing_clients > 0:
                metrics["updated_experts"] += 1
                metrics["evidence_clients"] += contributing_clients
                metrics["local_posteriors"] += local_posterior_count
            else:
                metrics["skipped_experts"] += 1
            expert_state.update(expert_params)

        if self.meta_update_mode == "closed_form_weighted":
            metrics["bayes_weighted_updated_experts"] = metrics["updated_experts"]
            metrics["bayes_weighted_skipped_experts"] = metrics["skipped_experts"]
        updated_bayes_state["round"] = int(updated_bayes_state.get("round", 0)) + 1
        metrics["bayes_aggregation_time_sec"] = round(time.perf_counter() - aggregate_start_time, 4)
        if self.meta_device.type == "cuda" and self.empty_cache_after_aggregation:
            torch.cuda.empty_cache()
        return {
            "expert_state": expert_state,
            "bayes_state": updated_bayes_state,
            "metrics": metrics,
        }

    def _aggregate_expert_group(
        self,
        layer_id,
        expert_id,
        expert_keys,
        global_state,
        updated_bayes_state,
        expert_evidence,
    ):
        expert_start_time = time.perf_counter()
        prior_state = self._get_or_init_prior_state(
            updated_bayes_state=updated_bayes_state,
            layer_id=layer_id,
            expert_id=expert_id,
            expert_keys=expert_keys,
            global_state=global_state,
        )
        client_payloads = self._collect_client_payloads(expert_evidence, layer_id, expert_id, expert_keys)
        if len(client_payloads) == 0:
            metric = self._build_expert_metric(
                layer_id=layer_id,
                expert_id=expert_id,
                prior_state=prior_state,
                meta_loss=None,
                contributing_clients=0,
                local_posterior_count=0,
                status="skipped_no_evidence",
                expert_keys=expert_keys,
                client_payloads=client_payloads,
            )
            metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
            return self._global_expert_params(global_state, expert_keys), 0, 0, metric

        prior_n0 = self._get_prior_n0(prior_state)
        if self.meta_update_mode == "closed_form_weighted":
            return self._aggregate_expert_group_closed_form_weighted(
                expert_keys=expert_keys,
                global_state=global_state,
                prior_state=prior_state,
                prior_n0=prior_n0,
                client_payloads=client_payloads,
                layer_id=layer_id,
                expert_id=expert_id,
                expert_start_time=expert_start_time,
            )

        optimized_mean_state, optimized_log_precision_state, optimized_log_n0, local_posterior_count, meta_loss = (
            self._optimize_expert_prior(
                expert_keys=expert_keys,
                global_state=global_state,
                prior_state=prior_state,
                prior_n0=prior_n0,
                client_payloads=client_payloads,
                layer_id=layer_id,
                expert_id=expert_id,
            )
        )
        if local_posterior_count <= 0:
            metric = self._build_expert_metric(
                layer_id=layer_id,
                expert_id=expert_id,
                prior_state=prior_state,
                meta_loss=None,
                contributing_clients=0,
                local_posterior_count=0,
                status="skipped_empty_terms",
                expert_keys=expert_keys,
                client_payloads=client_payloads,
            )
            metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
            return self._global_expert_params(global_state, expert_keys), 0, 0, metric

        aggregated_params = {
            key: optimized_mean_state[key].detach().cpu().to(dtype=global_state[key].dtype).clone()
            for key in expert_keys
        }
        if self.update_precision:
            for key in expert_keys:
                prior_state["log_precision_state"][key] = optimized_log_precision_state[key].detach().cpu()
        if self.update_strength:
            prior_state["log_n0"] = optimized_log_n0.detach().cpu()

        metric = self._build_expert_metric(
            layer_id=layer_id,
            expert_id=expert_id,
            prior_state=prior_state,
            meta_loss=meta_loss,
            contributing_clients=len(client_payloads),
            local_posterior_count=local_posterior_count,
            status="updated",
            optimized_log_precision_state=optimized_log_precision_state,
            optimized_log_n0=optimized_log_n0,
            expert_keys=expert_keys,
            client_payloads=client_payloads,
            global_state=global_state,
            optimized_mean_state=optimized_mean_state,
        )
        metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
        return aggregated_params, len(client_payloads), local_posterior_count, metric

    def _aggregate_expert_group_closed_form_weighted(
        self,
        expert_keys,
        global_state,
        prior_state,
        prior_n0,
        client_payloads,
        layer_id,
        expert_id,
        expert_start_time,
    ):
        def build_skipped_result(valid_client_payloads):
            expert_metric = self._build_expert_metric(
                layer_id=layer_id,
                expert_id=expert_id,
                prior_state=prior_state,
                meta_loss=None,
                contributing_clients=0,
                local_posterior_count=0,
                status="skipped",
                expert_keys=expert_keys,
                client_payloads=valid_client_payloads,
            )
            expert_metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
            return self._global_expert_params(global_state, expert_keys), 0, 0, expert_metric

        prior_n0 = prior_n0.detach().to(device=self.meta_device, dtype=torch.float32)
        if not torch.isfinite(prior_n0).all().item() or prior_n0.numel() != 1 or prior_n0.item() <= 0:
            return build_skipped_result([])

        prior_mean_state = collections.OrderedDict()
        prior_var_state = collections.OrderedDict()
        for key in expert_keys:
            prior_mean = global_state[key].detach().to(device=self.meta_device, dtype=torch.float32).clone()
            log_precision = prior_state.get("log_precision_state", {}).get(key)
            if (
                not torch.is_floating_point(prior_mean)
                or not torch.is_tensor(log_precision)
                or not torch.is_floating_point(log_precision)
                or log_precision.shape != prior_mean.shape
            ):
                return build_skipped_result([])
            log_precision = log_precision.detach().to(device=self.meta_device, dtype=torch.float32)
            prior_precision = _sanitize_precision_tensor(torch.exp(log_precision), self.config)
            if prior_precision is None:
                return build_skipped_result([])
            prior_var = _sanitize_var_tensor(1.0 / prior_precision, self.config)
            if prior_var is None:
                return build_skipped_result([])
            prior_mean_state[key] = prior_mean
            prior_var_state[key] = prior_var

        if self.weighted_precision_calibration == "median_target":
            calibration_payloads = []
            for payload in client_payloads:
                posterior = _compute_local_posterior_for_client(
                    local_mean_state=payload.get("mean_state"),
                    local_precision_state=payload.get("precision_state"),
                    prior_mean_state=prior_mean_state,
                    prior_var_state=prior_var_state,
                    n0=prior_n0,
                    config=self.config,
                )
                if posterior is not None:
                    calibration_payloads.append(payload)
            precision_scales, precision_calibration_diag = _build_shared_precision_calibration(
                client_payloads=calibration_payloads,
                expert_keys=expert_keys,
                config=self.config,
            )
        else:
            precision_scales, precision_calibration_diag = _build_shared_precision_calibration(
                client_payloads=[],
                expert_keys=expert_keys,
                config=self.config,
            )

        valid_clients = []
        for payload in client_payloads:
            local_precision_state = payload.get("precision_state")
            precision_clip_diag = None
            if self.weighted_precision_calibration in {
                "none",
                "median_target",
                }:
                if self.weighted_diag:
                    precision_clip_diag = {
                        "precision_clip_max_count": 0,
                        "precision_clip_min_count": 0,
                        "precision_total_count": 0,
                    }
                local_precision_state = _apply_shared_precision_calibration(
                    local_precision_state=local_precision_state,
                    expert_keys=expert_keys,
                    scales=precision_scales,
                    config=self.config,
                    precision_clip_diag=precision_clip_diag,
                    replace_nonfinite=self.weighted_precision_calibration != "median_target",
                )
            posterior = _compute_local_posterior_for_client(
                local_mean_state=payload.get("mean_state"),
                local_precision_state=local_precision_state,
                prior_mean_state=prior_mean_state,
                prior_var_state=prior_var_state,
                n0=prior_n0,
                config=self.config,
            )
            if posterior is None:
                continue
            m_star_state, v_star_state, sanitized_precision_state = posterior
            score = _compute_evidence_score_scalar(
                local_mean_state=payload.get("mean_state"),
                local_precision_state=sanitized_precision_state,
                prior_mean_state=prior_mean_state,
                prior_var_state=prior_var_state,
                m_star_state=m_star_state,
                v_star_state=v_star_state,
                n0=prior_n0,
                config=self.config,
            )
            if score is None:
                continue
            valid_clients.append({
                "payload": payload,
                "m_star_state": m_star_state,
                "raw_precision_state": payload.get("precision_state"),
                "sanitized_precision_state": sanitized_precision_state,
                "score": score,
                "precision_clip_diag": precision_clip_diag,
            })

        beta = _scores_to_beta([client["score"] for client in valid_clients], self.config)
        if beta is None:
            return build_skipped_result([client["payload"] for client in valid_clients])

        if self.v0_update_mode == "precision_ema":
            rho = float(self.weighted_var_rho)
            if not math.isfinite(rho):
                return build_skipped_result([client["payload"] for client in valid_clients])

        new_mean_state = collections.OrderedDict()
        new_log_precision_state = collections.OrderedDict()
        with torch.no_grad():
            for key in expert_keys:
                new_mean = torch.zeros_like(prior_mean_state[key])
                a_hat = torch.zeros_like(prior_mean_state[key]) if self.v0_update_mode == "precision_ema" else None
                for client_beta, client in zip(beta, valid_clients):
                    new_mean += client_beta * client["m_star_state"][key]
                    if a_hat is not None:
                        a_hat += client_beta * client["sanitized_precision_state"][key]
                if not torch.isfinite(new_mean).all().item():
                    return build_skipped_result([client["payload"] for client in valid_clients])

                new_mean_state[key] = new_mean
                if self.v0_update_mode == "fixed":
                    log_precision = prior_state.get("log_precision_state", {}).get(key)
                    if not torch.is_tensor(log_precision) or not torch.is_floating_point(log_precision):
                        return build_skipped_result([client["payload"] for client in valid_clients])
                    new_log_precision_state[key] = log_precision.detach().to(
                        device=self.meta_device,
                        dtype=torch.float32,
                    )
                    continue

                if a_hat is None or not torch.isfinite(a_hat).all().item():
                    return build_skipped_result([client["payload"] for client in valid_clients])
                p_old = prior_n0 / prior_var_state[key]
                p_new = (1.0 - rho) * p_old + rho * a_hat
                if not torch.isfinite(p_new).all().item() or (p_new <= 0).any().item():
                    return build_skipped_result([client["payload"] for client in valid_clients])
                v_new = _sanitize_var_tensor(prior_n0 / p_new, self.config)
                if v_new is None:
                    return build_skipped_result([client["payload"] for client in valid_clients])
                precision_new = _sanitize_precision_tensor(1.0 / v_new, self.config)
                if precision_new is None:
                    return build_skipped_result([client["payload"] for client in valid_clients])
                new_log_precision_state[key] = torch.log(precision_new)

        aggregated_params = {
            key: new_mean_state[key].detach().cpu().to(dtype=global_state[key].dtype).clone()
            for key in expert_keys
        }
        if self.v0_update_mode == "precision_ema":
            for key in expert_keys:
                prior_state["log_precision_state"][key] = new_log_precision_state[key].detach().cpu()

        local_posterior_count = len(valid_clients) * len(expert_keys)
        expert_metric = self._build_expert_metric(
            layer_id=layer_id,
            expert_id=expert_id,
            prior_state=prior_state,
            meta_loss=None,
            contributing_clients=len(valid_clients),
            local_posterior_count=local_posterior_count,
            status="updated",
            optimized_log_precision_state=new_log_precision_state,
            optimized_log_n0=prior_state.get("log_n0"),
            expert_keys=expert_keys,
            client_payloads=[client["payload"] for client in valid_clients],
            global_state=global_state,
            optimized_mean_state=new_mean_state,
        )
        if self.weighted_diag:
            expert_metric.update(self._build_closed_form_weighted_diag(
                beta=beta,
                valid_clients=valid_clients,
                prior_var_state=prior_var_state,
                prior_mean_state=prior_mean_state,
                new_mean_state=new_mean_state,
                prior_n0=prior_n0,
                precision_calibration_diag=precision_calibration_diag,
            ))
        expert_metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
        return aggregated_params, len(valid_clients), local_posterior_count, expert_metric

    def _build_closed_form_weighted_diag(
        self,
        beta,
        valid_clients,
        prior_var_state,
        prior_mean_state,
        new_mean_state,
        prior_n0,
        precision_calibration_diag,
    ):
        def summarize_tensors(tensors, include_std):
            value_sum = 0.0
            value_square_sum = 0.0
            value_count = 0
            value_min = None
            value_max = None
            for value in tensors:
                if not torch.is_tensor(value):
                    continue
                value = value.detach().double().reshape(-1)
                value = value[torch.isfinite(value)]
                if value.numel() == 0:
                    continue
                value_sum += float(value.sum().item())
                value_square_sum += float(value.square().sum().item())
                value_count += value.numel()
                tensor_min = float(value.min().item())
                tensor_max = float(value.max().item())
                value_min = tensor_min if value_min is None else min(value_min, tensor_min)
                value_max = tensor_max if value_max is None else max(value_max, tensor_max)
            summary = {"mean": None, "min": None, "max": None}
            if include_std:
                summary["std"] = None
            if value_count == 0:
                return summary
            mean = value_sum / value_count
            summary.update({"mean": mean, "min": value_min, "max": value_max})
            if include_std:
                variance = max(value_square_sum / value_count - mean * mean, 0.0)
                summary["std"] = math.sqrt(variance)
            return summary

        beta_tensor = torch.tensor(beta, dtype=torch.float64)
        score_tensor = torch.tensor([client["score"] for client in valid_clients], dtype=torch.float64)
        beta_entropy = float(
            -(beta_tensor * torch.log(beta_tensor.clamp(min=self.weighted_eps))).sum().item()
        )
        raw_precision_stats = summarize_tensors(
            [
                client["raw_precision_state"].get(key)
                for client in valid_clients
                if isinstance(client.get("raw_precision_state"), dict)
                for key in client["sanitized_precision_state"]
            ],
            include_std=True,
        )
        precision_stats = summarize_tensors(
            [
                client["sanitized_precision_state"][key]
                for client in valid_clients
                for key in client["sanitized_precision_state"]
            ],
            include_std=True,
        )
        gamma0_tensors = []
        for key, prior_var in prior_var_state.items():
            n0_tensor = prior_n0.to(device=prior_var.device, dtype=prior_var.dtype)
            gamma0_tensors.append(n0_tensor / prior_var)
        gamma0_stats = summarize_tensors(gamma0_tensors, include_std=True)

        alpha_tensors = []
        for client in valid_clients:
            for key, cal_precision in client["sanitized_precision_state"].items():
                prior_var = prior_var_state.get(key)
                if not torch.is_tensor(prior_var) or not torch.is_tensor(cal_precision):
                    continue
                cal_precision = cal_precision.detach()
                prior_var = prior_var.detach().to(
                    device=cal_precision.device,
                    dtype=cal_precision.dtype,
                )
                n0_tensor = prior_n0.to(device=cal_precision.device, dtype=cal_precision.dtype)
                gamma0 = n0_tensor / prior_var
                alpha_tensors.append(cal_precision / (cal_precision + gamma0))
        alpha_stats = summarize_tensors(alpha_tensors, include_std=True)

        precision_clip_diags = [
            client["precision_clip_diag"]
            for client in valid_clients
            if client["precision_clip_diag"] is not None
        ]
        precision_clip_stats = {}
        if precision_clip_diags:
            precision_clip_max_count = sum(
                diag["precision_clip_max_count"] for diag in precision_clip_diags
            )
            precision_clip_min_count = sum(
                diag["precision_clip_min_count"] for diag in precision_clip_diags
            )
            precision_total_count = sum(
                diag["precision_total_count"] for diag in precision_clip_diags
            )
            precision_clip_count = precision_clip_max_count + precision_clip_min_count
            if precision_total_count > 0:
                precision_clip_max_frac = precision_clip_max_count / precision_total_count
                precision_clip_min_frac = precision_clip_min_count / precision_total_count
                precision_clip_frac = precision_clip_count / precision_total_count
            else:
                precision_clip_max_frac = 0.0
                precision_clip_min_frac = 0.0
                precision_clip_frac = 0.0
            precision_clip_stats = {
                "precision_clip_frac": precision_clip_frac,
                "precision_clip_max_frac": precision_clip_max_frac,
                "precision_clip_min_frac": precision_clip_min_frac,
                "precision_clip_count": precision_clip_count,
                "precision_total_count": precision_total_count,
            }
        prior_var_stats = summarize_tensors(prior_var_state.values(), include_std=False)
        mean_update_square_sum = sum(
            float((new_mean_state[key] - prior_mean_state[key]).double().square().sum().item())
            for key in new_mean_state
        )
        return {
            "v0_update_mode": self.v0_update_mode,
            "beta_min": float(beta_tensor.min().item()),
            "beta_max": float(beta_tensor.max().item()),
            "beta_mean": float(beta_tensor.mean().item()),
            "beta_entropy": beta_entropy,
            "beta_num_clients": len(beta),
            "score_min": float(score_tensor.min().item()),
            "score_max": float(score_tensor.max().item()),
            "score_mean": float(score_tensor.mean().item()),
            "score_std": float(score_tensor.std(unbiased=False).item()),
            "precision_mean": precision_stats["mean"],
            "precision_std": precision_stats["std"],
            "precision_min": precision_stats["min"],
            "precision_max": precision_stats["max"],
            "raw_A_mean": raw_precision_stats["mean"],
            "raw_A_std": raw_precision_stats["std"],
            "raw_A_min": raw_precision_stats["min"],
            "raw_A_max": raw_precision_stats["max"],
            "cal_A_mean": precision_stats["mean"],
            "cal_A_std": precision_stats["std"],
            "cal_A_min": precision_stats["min"],
            "cal_A_max": precision_stats["max"],
            "gamma0_mean": gamma0_stats["mean"],
            "gamma0_std": gamma0_stats["std"],
            "gamma0_min": gamma0_stats["min"],
            "gamma0_max": gamma0_stats["max"],
            "alpha_mean": alpha_stats["mean"],
            "alpha_std": alpha_stats["std"],
            "alpha_min": alpha_stats["min"],
            "alpha_max": alpha_stats["max"],
            "prior_var_mean": prior_var_stats["mean"],
            "prior_var_min": prior_var_stats["min"],
            "prior_var_max": prior_var_stats["max"],
            "mean_update_norm": math.sqrt(mean_update_square_sum),
            **precision_clip_stats,
            **precision_calibration_diag,
        }

    def _optimize_expert_prior(
        self,
        expert_keys,
        global_state,
        prior_state,
        prior_n0,
        client_payloads,
        layer_id,
        expert_id,
    ):
        prior_mean_params = collections.OrderedDict()
        log_precision_params = collections.OrderedDict()
        optim_params = []
        for key in expert_keys:
            reference_tensor = global_state[key].detach().to(device=self.meta_device, dtype=torch.float32).clone()
            mean_param = torch.nn.Parameter(reference_tensor.clone())
            prior_mean_params[key] = mean_param
            optim_params.append(mean_param)

            log_precision = prior_state.get("log_precision_state", {}).get(key)
            if log_precision is None or not torch.is_floating_point(log_precision):
                log_precision = torch.full_like(reference_tensor, fill_value=math.log(self.gamma0_init))
            else:
                log_precision = log_precision.detach().to(device=self.meta_device, dtype=torch.float32).clone()
            log_precision_param = torch.nn.Parameter(log_precision, requires_grad=self.update_precision)
            log_precision_params[key] = log_precision_param
            if self.update_precision:
                optim_params.append(log_precision_param)

        log_n0_value = prior_state.get("log_n0")
        if log_n0_value is None:
            log_n0_value = torch.tensor(math.log(max(float(prior_n0.item()), self.min_precision)))
        log_n0_value = log_n0_value.detach().to(device=self.meta_device, dtype=torch.float32).clone()
        log_n0_param = torch.nn.Parameter(log_n0_value, requires_grad=self.update_strength)
        if self.update_strength:
            optim_params.append(log_n0_param)

        optimizer = torch.optim.Adam(optim_params, lr=self.meta_lr)
        last_finite_loss = None
        local_posterior_count = 0
        for _ in range(self.meta_steps):
            optimizer.zero_grad()
            meta_loss, local_posterior_count = self._compute_expert_meta_loss(
                expert_keys=expert_keys,
                prior_mean_params=prior_mean_params,
                log_precision_params=log_precision_params,
                log_n0_param=log_n0_param,
                client_payloads=client_payloads,
            )
            if local_posterior_count <= 0:
                break
            if not torch.isfinite(meta_loss).item():
                print(
                    "[ExpertBayesMetaAggregator] warning: non-finite meta_loss "
                    f"layer={layer_id} expert={expert_id}; keeping last finite parameters"
                )
                break
            meta_loss.backward()
            optimizer.step()
            self._project_meta_params(log_precision_params, log_n0_param)
            last_finite_loss = float(meta_loss.detach().cpu().item())

        return prior_mean_params, log_precision_params, log_n0_param, local_posterior_count, last_finite_loss

    def _compute_expert_meta_loss(
        self,
        expert_keys,
        prior_mean_params,
        log_precision_params,
        log_n0_param,
        client_payloads,
    ):
        zero = next(iter(prior_mean_params.values())).new_tensor(0.0)
        client_losses = []
        local_posterior_count = 0
        prior_n0 = torch.exp(log_n0_param).clamp(min=self.min_precision, max=self.max_n0)
        for payload in client_payloads:
            client_loss = zero
            has_local_terms = False
            for key in expert_keys:
                local_mean = payload["mean_state"].get(key)
                local_precision = payload["precision_state"].get(key)
                if local_mean is None or local_precision is None:
                    continue
                prior_mean = prior_mean_params[key]
                local_mean = local_mean.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
                local_precision = local_precision.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
                prior_precision = torch.exp(log_precision_params[key]).clamp(
                    min=self.min_precision,
                    max=self.max_precision,
                )
                posterior_mean, _, posterior_precision = compute_optimal_local_posterior(
                    local_mean=local_mean,
                    local_precision=local_precision,
                    prior_mean=prior_mean,
                    prior_precision=prior_precision,
                    prior_n0=prior_n0,
                    min_precision=self.min_precision,
                )
                fit_term, regularizer = compute_quadratic_meta_terms(
                    local_mean=local_mean,
                    local_precision=local_precision,
                    prior_mean=prior_mean,
                    prior_precision=prior_precision,
                    prior_n0=prior_n0,
                    posterior_mean=posterior_mean,
                    posterior_precision=posterior_precision,
                    min_precision=self.min_precision,
                )
                client_loss = client_loss + fit_term + 0.5 * regularizer
                has_local_terms = True
                local_posterior_count += 1
            if has_local_terms:
                client_losses.append(client_loss)
        if len(client_losses) == 0:
            return zero, 0
        return torch.stack(client_losses).mean(), local_posterior_count

    def _project_meta_params(self, log_precision_params, log_n0_param):
        log_min_precision = math.log(self.min_precision)
        log_max_precision = math.log(self.max_precision)
        log_max_n0 = math.log(self.max_n0)
        with torch.no_grad():
            for log_precision_param in log_precision_params.values():
                log_precision_param.clamp_(min=log_min_precision, max=log_max_precision)
            log_n0_param.clamp_(min=log_min_precision, max=log_max_n0)

    def _collect_client_payloads(self, expert_evidence, layer_id, expert_id, expert_keys):
        payloads = []
        for client_evidence in expert_evidence:
            evidence = get_client_expert_evidence(client_evidence, layer_id, expert_id)
            if evidence is None:
                continue
            mean_state = evidence.get("mean_state")
            precision_state = evidence.get("precision_state")
            if not isinstance(mean_state, dict) or not isinstance(precision_state, dict):
                continue
            if not any(key in mean_state and key in precision_state for key in expert_keys):
                continue
            payloads.append({
                "num_batches": int(evidence.get("num_batches", 0)),
                "mean_state": mean_state,
                "precision_state": precision_state,
            })
        return payloads

    def _get_or_init_prior_state(self, updated_bayes_state, layer_id, expert_id, expert_keys, global_state):
        expert_state = get_bayes_expert_state(updated_bayes_state, layer_id, expert_id)
        if expert_state is not None:
            return expert_state
        layer_state = updated_bayes_state.setdefault("experts", {}).setdefault(str(layer_id), {})
        log_gamma0 = math.log(self.gamma0_init)
        log_precision_state = {}
        for key in expert_keys:
            value = global_state[key].detach().cpu()
            log_precision_state[key] = torch.full_like(value, fill_value=log_gamma0)
        layer_state[str(expert_id)] = {
            "log_precision_state": log_precision_state,
            "log_n0": torch.tensor(math.log(self.n0_init), dtype=torch.float32),
        }
        return layer_state[str(expert_id)]

    def _get_prior_n0(self, prior_state):
        log_n0 = prior_state.get("log_n0")
        if log_n0 is None:
            return torch.tensor(self.n0_init, dtype=torch.float32)
        return torch.exp(log_n0.detach().cpu().float()).clamp(min=self.min_precision)

    def _global_expert_params(self, global_state, expert_keys):
        return {key: global_state[key].detach().cpu().clone() for key in expert_keys}

    def _summarize_client_payloads(self, client_payloads, expert_keys):
        precision_values = []
        for payload in client_payloads:
            for key in expert_keys:
                value = payload["precision_state"].get(key)
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    precision_values.append(value)
        precision_summary = _summarize_tensors(precision_values)
        return {
            "num_batches_total": sum(int(payload.get("num_batches", 0)) for payload in client_payloads),
            "local_precision_mean": precision_summary["mean"],
            "local_precision_min": precision_summary["min"],
            "local_precision_max": precision_summary["max"],
        }

    def _summarize_mean_update(self, global_state, optimized_mean_state, expert_keys):
        if global_state is None or optimized_mean_state is None:
            return {"param_delta_rel": None}
        delta_sq = 0.0
        prior_sq = 0.0
        for key in expert_keys:
            prior_value = global_state[key].detach().cpu().float()
            updated_value = optimized_mean_state[key].detach().cpu().float()
            delta_sq += float((updated_value - prior_value).square().sum().item())
            prior_sq += float(prior_value.square().sum().item())
        denom = max(math.sqrt(max(prior_sq, 0.0)), 1.0e-12)
        return {"param_delta_rel": round(math.sqrt(max(delta_sq, 0.0)) / denom, 6)}

    def _build_expert_metric(
        self,
        layer_id,
        expert_id,
        prior_state,
        meta_loss,
        contributing_clients,
        local_posterior_count,
        status,
        optimized_log_precision_state=None,
        optimized_log_n0=None,
        expert_keys=None,
        client_payloads=None,
        global_state=None,
        optimized_mean_state=None,
    ):
        log_precision_state = optimized_log_precision_state or prior_state.get("log_precision_state", {})
        gamma_summary = _summarize_tensors([torch.exp(value.detach().cpu().float()) for value in log_precision_state.values()])
        log_n0 = optimized_log_n0 if optimized_log_n0 is not None else prior_state.get("log_n0")
        n0 = self.n0_init if log_n0 is None else float(torch.exp(log_n0.detach().cpu().float()).item())
        expert_keys = expert_keys or []
        client_payloads = client_payloads or []
        metric = {
            "status": status,
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "clients": int(contributing_clients),
            "local_posteriors": int(local_posterior_count),
            "meta_loss": None if meta_loss is None else round(float(meta_loss), 6),
            "n0": round(float(n0), 6),
            "precision_mean": gamma_summary["mean"],
            "precision_min": gamma_summary["min"],
            "precision_max": gamma_summary["max"],
        }
        metric.update(self._summarize_client_payloads(client_payloads, expert_keys))
        metric.update(self._summarize_mean_update(global_state, optimized_mean_state, expert_keys))
        return metric


def aggregate_expert_bayes(global_state, client_states, client_bayes_evidence, bayes_state, config):
    aggregator = ExpertBayesMetaAggregator(config)
    return aggregator.aggregate(
        client_updates=client_states,
        global_state=global_state,
        expert_evidence=client_bayes_evidence,
        bayes_state=bayes_state,
    )


def aggregate_split_model(
    global_model,
    client_states,
    client_samples,
    non_expert_agg_method="uniform",
    expert_agg_method="uniform",
    agg_method="decoupled_moe",
    client_bayes_evidence=None,
    bayes_state=None,
    bayes_config=None,
):
    global_state = global_model.state_dict()
    expert_keys, non_expert_keys = split_state_keys(global_state)
    agg_method = str(agg_method or "decoupled_moe").lower()
    effective_expert_agg_method = (
        "expert_bayes_meta" if agg_method == "expert_bayes_meta" else str(expert_agg_method).lower()
    )

    new_state = {}
    new_state.update(
        _aggregate_keys_by_method(
            global_state,
            client_states,
            client_samples,
            non_expert_keys,
            non_expert_agg_method,
        )
    )

    if effective_expert_agg_method == "expert_bayes_meta":
        bayes_output = aggregate_expert_bayes(
            global_state=global_state,
            client_states=client_states,
            client_bayes_evidence=client_bayes_evidence,
            bayes_state=bayes_state,
            config=bayes_config or {},
        )
        new_state.update(bayes_output["expert_state"])
        if set(new_state.keys()) != set(global_state.keys()):
            raise ValueError("Aggregated state keys do not match global_state keys")
        return {
            "model_state": {key: new_state[key] for key in global_state.keys()},
            "bayes_state": bayes_output["bayes_state"],
            "metrics": bayes_output["metrics"],
        }

    new_state.update(
        _aggregate_keys_by_method(
            global_state,
            client_states,
            client_samples,
            expert_keys,
            effective_expert_agg_method,
        )
    )

    if set(new_state.keys()) != set(global_state.keys()):
        raise ValueError("Aggregated state keys do not match global_state keys")

    return {key: new_state[key] for key in global_state.keys()}
