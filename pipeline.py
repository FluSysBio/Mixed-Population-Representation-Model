"""End-to-end training routine"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


from .constants import AA_LIST
from .data import expand_replicates, load_mn_data
from .encoder import load_encoder
from .evaluate import evaluate_split, mixture_response_grid, run_cross_validation, token_attention_report
from .features import CTX_COLS, RATIO_FEATURE_NAMES, build_feature_bank, build_row_features, train_only_context_features
from .inference import MPATPredictor, parse_ratio, save_bundle
from .losses import apply_weight_map, compute_sample_weights
from .splits import make_splits
from .structure import build_pssm, build_structure
from .train import train_ensemble
from .utils import json_safe, read_fasta, save_json, set_seed


def run_training(args) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    print(f"\n{'='*78}\n  MIXTURE -> TITER PIPELINE   |  device: {device}\n{'='*78}")

    # ---- 1. data ---------------------------------------------------------
    if getattr(args, "cv_epochs", None) is None:
        args.cv_epochs = args.epochs

    df = load_mn_data(args.data)
    print(f"[data] {len(df)} conditions | {df['virus_strain'].nunique()} viruses x "
          f"{df['serum_strain'].nunique()} sera x {df['ratio_group'].nunique()} mixtures")
    print(f"[data] log2 titer range {df['log2_titer'].min():.2f}-{df['log2_titer'].max():.2f} | "
          f"{int((df['censor_frac'] > 0).sum())} conditions contain a censored (<10) replicate")

    vseqs = read_fasta(args.virus_fasta)
    sseqs = read_fasta(args.serum_fasta) if getattr(args, "serum_fasta", None) else {}
    all_lengths = [len(s) for s in list(vseqs.values()) + list(sseqs.values())]
    graph_len = args.graph_len or int(max(all_lengths))

    # ---- 2. public-data priors ------------------------------------------
    adj, structure_info = build_structure(args.pdb, args.pdb_chain, graph_len,
                                          args.distance_cutoff)
    use_pssm = args.gisaid_fasta is not None
    if use_pssm:
        public = read_fasta(args.gisaid_fasta, max_sequences=args.max_pssm_seqs, seed=args.seed)
        pssm_logodds, pssm_entropy = build_pssm(list(public.values()), graph_len)
        print(f"[prior] PSSM built from {len(public)} public HA sequences")
    else:
        pssm_logodds = np.zeros((graph_len, len(AA_LIST)), np.float32)
        pssm_entropy = np.zeros(graph_len, np.float32)

    encoder, encoder_cfg = load_encoder(args.pretrained_encoder, device)
    if encoder_cfg.get("graph_len", graph_len) != graph_len:
        graph_len = encoder_cfg["graph_len"]
        adj, structure_info = build_structure(args.pdb, args.pdb_chain, graph_len,
                                              args.distance_cutoff)
        if use_pssm:
            pssm_logodds, pssm_entropy = build_pssm(list(public.values()), graph_len)
    print(f"[prior] encoder loaded ({encoder_cfg}) | graph_len={graph_len}")

    bank = build_feature_bank(df, vseqs, sseqs, encoder, adj, structure_info, graph_len,
                              pssm_logodds, pssm_entropy, device, use_pssm,
                              serum_mode=args.serum_mode)
    print(f"[feat] seq_dim={bank.seq_dim} pssm_dim={bank.pssm_dim} "
          f"static_pair_dim={bank.static_pair_dim}")

    # ---- 3. cross-validation protocols (optional) ------------------------
    cv_summary = {}
    if args.cv != "none":
        cv_summary = run_cross_validation(df, bank, args, device, out_dir)
        save_json(cv_summary, out_dir / "cross_validation" / "cv_summary.json")
        if args.cv_only:
            print(f"\n[done] CV-only run | artefacts -> {out_dir.resolve()}\n")
            return

    # ---- 4. splits -------------------------------------------------------
    train_df, val_df, test_df = make_splits(df, args.split_mode, args.test_size,
                                            args.val_size, args.seed)
    print(f"[split] mode={args.split_mode} | train {len(train_df)} / val {len(val_df)} "
          f"/ test {len(test_df)}")

    # context features are derived from the training split only
    train_df = train_only_context_features(train_df, train_df)
    val_df = train_only_context_features(train_df, val_df)
    test_df = train_only_context_features(train_df, test_df)

    train_fit = train_df.copy()             # collapsed reference for context/bundle
    if args.use_replicates:
        train_df = train_only_context_features(train_fit, expand_replicates(train_fit))
        print(f"[split] replicate expansion: train rows {len(train_fit)} -> {len(train_df)}")

    # ---- 5. featurise + scale (fit on train only) ------------------------
    X_train_raw = build_row_features(train_df, bank)
    scaler = StandardScaler().fit(X_train_raw)
    scaler.scale_[scaler.scale_ == 0] = 1.0
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(build_row_features(val_df, bank)).astype(np.float32)
    X_test = scaler.transform(build_row_features(test_df, bank)).astype(np.float32)
    feature_names = (list(bank.pair_names) + list(RATIO_FEATURE_NAMES)
                     + [c for c in CTX_COLS if c in train_df.columns])
    print(f"[feat] engineered feature vector: {X_train.shape[1]} dims")

    w_train, wmap = compute_sample_weights(train_df, args.weight_scheme, args.max_titer_weight)
    w_val = apply_weight_map(val_df, wmap)

    # ---- 6. train --------------------------------------------------------
    print(f"\n[train] ensemble of {args.n_seeds} BiLSTM models")
    models, histories, val_scores = train_ensemble(train_df, val_df, bank, X_train, X_val,
                                                   w_train, w_val, args, device)
    save_json({"histories": histories, "best_val_monitor": val_scores,
               "monitor": args.monitor}, out_dir / "training_history.json")

    # ---- 7. evaluate -----------------------------------------------------
    print("\n[eval] ensemble performance")
    metrics = {
        "train": evaluate_split("train", train_df, models, bank, X_train, device, args, out_dir),
        "val": evaluate_split("val", val_df, models, bank, X_val, device, args, out_dir),
        "test": evaluate_split("test", test_df, models, bank, X_test, device, args, out_dir),
    }
    metrics["per_model_val_monitor"] = val_scores
    metrics["config"] = json_safe(vars(args))
    metrics["dataset"] = {
        "n_conditions": int(len(df)), "n_train_rows": int(len(train_df)),
        "n_val": int(len(val_df)), "n_test": int(len(test_df)),
        "replicate_expansion": bool(args.use_replicates),
        "viruses": sorted(df["virus_strain"].unique().tolist()),
        "sera": sorted(df["serum_strain"].unique().tolist()),
        "mixtures": sorted(df["ratio_string_KTI"].unique().tolist())
        if "ratio_string_KTI" in df.columns else [],
    }
    metrics["token_attention"] = token_attention_report(models, test_df, bank, X_test, device)
    if cv_summary:
        metrics["cross_validation"] = cv_summary
    save_json(metrics, out_dir / "metrics_all.json")

    # ---- 8. mixture-response surface + epistasis -------------------------
    if not args.skip_grid:
        grid = mixture_response_grid(models, bank, train_fit, device, args, args.grid_step)
        grid.to_csv(out_dir / "mixture_response_grid.csv", index=False)
        eps = grid["epistasis_log2"].abs()
        print(f"[grid] simplex surface saved ({len(grid)} points) | "
              f"mean |epistasis| = {eps.mean():.3f} log2, max = {eps.max():.3f} log2")
        metrics["epistasis_summary"] = {
            "mean_abs_epistasis_log2": float(eps.mean()),
            "max_abs_epistasis_log2": float(eps.max()),
            "frac_beyond_half_dilution": float((eps > 0.5).mean()),
        }
        save_json(metrics, out_dir / "metrics_all.json")

    # ---- 9. persist ------------------------------------------------------
    encoder_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    save_bundle(out_dir / "model_bundle.pt", models, bank, scaler, encoder_state,
                encoder_cfg, adj, structure_info, pssm_logodds, pssm_entropy,
                train_fit, args, metrics, feature_names)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), out_dir / f"model_seed{i}.pt")
    train_fit.to_csv(out_dir / "split_train.csv", index=False)
    val_df.to_csv(out_dir / "split_val.csv", index=False)
    test_df.to_csv(out_dir / "split_test.csv", index=False)
    save_json({"feature_names": feature_names}, out_dir / "feature_names.json")

    t = metrics["test"]
    print(f"\n{'='*78}\n  HELD-OUT TEST (n={t['n']}): "
          f"MAE {t['mae_log2']:.3f} log2 | RMSE {t['rmse_log2']:.3f} | R2 {t['r2']:.3f} | "
          f"within-1-dilution {t['within_1_dilution']*100:.1f}%")
    if cv_summary:
        for name, row in cv_summary.items():
            print(f"  {name:26s}: MAE {row['MAE_log2']:.3f} | R2 {row['R2']:6.3f} | "
                  f"within-1 {row['within_1_dilution']*100:.1f}%")
    print(f"  artefacts -> {out_dir.resolve()}\n{'='*78}\n")


def run_prediction(args) -> None:
    p = MPATPredictor(args.bundle, device="cpu" if args.cpu else None,
                              virus_fasta=args.virus_fasta,
                              serum_fasta=getattr(args, "serum_fasta", None))
    if args.query_csv:
        q = pd.read_csv(args.query_csv)
        queries = [(r["virus_strain"] if "virus_strain" in q.columns else r["strain"],
                    r["serum_strain"] if "serum_strain" in q.columns else r["serum"],
                    parse_ratio(r["ratio"]) if "ratio" in q.columns
                    else (r["K_fraction"], r["T_fraction"], r["I_fraction"]))
                   for _, r in q.iterrows()]
        out = p.predict_many(queries)
    elif args.simplex:
        out = p.predict_simplex(args.virus, args.serum, args.grid_step)
    else:
        out = p.predict_many([(args.virus, args.serum, parse_ratio(args.ratio))])

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"saved -> {args.out_csv}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(out.head(30).to_string(index=False))
