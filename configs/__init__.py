"""Configuration defaults for the lightweight training entrypoint."""


DEFAULT_BAYES_CONFIG = {
    "resume": False,
    "resume_checkpoint_path": None,
    "bayes_precision_source": "sgld_variance",
    "bayes_sgld_fit_mode": "adam_noise",
    "bayes_precision_mode": "floor_inverse",
    "bayes_sgld_timing": "after_train",
    "bayes_evidence_mean_source": "sgld_mean",
    "bayes_sgld_steps": 10,
    "bayes_sgld_burnin": 5,
    "bayes_sgld_lr": 0.00005,
    "bayes_sgld_var_floor": 0.0,
    "bayes_precision_eps": 1.0e-12,
    "bayes_evidence_batches": 4,
    "bayes_min_expert_tokens": 128,
    "bayes_meta_update_mode": "optimizer",
    "bayes_meta_steps": 2,
    "bayes_meta_lr": 0.0005,
    "bayes_gamma0_init": 1.0,
    "bayes_n0_init": 1.0,
    "bayes_update_precision": True,
    "bayes_update_strength": True,
    "bayes_meta_device": "auto",
    "bayes_cache_device": "cuda",
    "bayes_empty_cache_after_aggregation": False,
    "bayes_empty_cache_after_client_evidence": False,
    "bayes_sgld_movement_diag": False,
    "bayes_sgld_movement_diag_detail": False,
    "bayes_sgld_stuck_rel_threshold": 1.0e-5,
    "bayes_sgld_stuck_abs_threshold": 1.0e-7,
    "bayes_weighted_score_tau": 0.5,
    "bayes_weighted_score_clip": 3.0,
    "bayes_weighted_var_rho": 0.05,
    "bayes_weighted_eps": 1.0e-8,
    "bayes_weighted_min_valid_clients": 2,
    "bayes_weighted_precision_min": 1.0e-8,
    "bayes_weighted_precision_max": 1.0e8,
    "bayes_precision_calibration_mode": "none",
    "bayes_weighted_precision_target": 100.0,
    "bayes_v0_update_mode": "precision_ema",
    "bayes_weighted_var_min": 1.0e-8,
    "bayes_weighted_var_max": 1.0e8,
    "bayes_weighted_diag": False,
}


def apply_config_defaults(cfg):
    cfg = dict(cfg or {})
    cfg.setdefault("agg_method", "decoupled_moe")

    if str(cfg["agg_method"]).lower() == "expert_bayes_meta":
        cfg["expert_agg_method"] = "expert_bayes_meta"
        cfg.setdefault("non_expert_agg_method", "sample_weighted")
    else:
        cfg.setdefault("non_expert_agg_method", "uniform")
        cfg.setdefault("expert_agg_method", "uniform")

    for key, value in DEFAULT_BAYES_CONFIG.items():
        cfg.setdefault(key, value)

    _validate_bayes_defaults(cfg)
    return cfg


def _validate_bayes_defaults(cfg):
    if str(cfg["bayes_precision_source"]).lower() != "sgld_variance":
        raise ValueError("bayes_precision_source now only supports: sgld_variance")
    if str(cfg["bayes_precision_mode"]).lower() != "floor_inverse":
        raise ValueError("bayes_precision_mode now only supports: floor_inverse")
    if str(cfg["bayes_sgld_fit_mode"]).lower() not in {"adam_noise", "sgd_noise"}:
        raise ValueError("bayes_sgld_fit_mode must be one of: adam_noise, sgd_noise")
    if str(cfg["bayes_meta_update_mode"]).lower() not in {"optimizer", "closed_form_weighted"}:
        raise ValueError(
            "bayes_meta_update_mode must be one of: optimizer, closed_form_weighted"
        )

    precision_calibration = cfg.get("bayes_precision_calibration_mode")
    if precision_calibration is None:
        precision_calibration = cfg.get("bayes_weighted_precision_calibration", "none")
    precision_calibration = str(precision_calibration).lower()
    if precision_calibration not in {"none", "median_target"}:
        raise ValueError(
            "bayes_precision_calibration_mode must be one of: none, median_target"
        )

    v0_update_mode = str(cfg.get("bayes_v0_update_mode", "precision_ema")).lower()
    if v0_update_mode not in {"precision_ema", "fixed"}:
        raise ValueError("bayes_v0_update_mode must be one of: precision_ema, fixed")

    sgld_timing = str(cfg["bayes_sgld_timing"]).lower()
    if sgld_timing not in {"after_train", "before_train"}:
        raise ValueError("bayes_sgld_timing must be one of: after_train, before_train")

    evidence_mean_source = str(cfg["bayes_evidence_mean_source"]).lower()
    if evidence_mean_source not in {"sgld_mean", "train_final"}:
        raise ValueError(
            "bayes_evidence_mean_source must be one of: sgld_mean, train_final"
        )
    if sgld_timing == "after_train" and evidence_mean_source == "train_final":
        raise ValueError(
            "bayes_evidence_mean_source=train_final currently requires "
            "bayes_sgld_timing=before_train"
        )
