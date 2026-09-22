"""Command-line interface."""

from __future__ import annotations

import argparse
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from .encoder import pretrain_encoder
from .pipeline import run_prediction, run_training


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="H3N2 HA mixture -> MN titer prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    # ---- pretrain --------------------------------------------------------
    pt = sub.add_parser("pretrain", help="masked-LM pre-training on public HA sequences")
    pt.add_argument("--gisaid_fasta", required=True)
    pt.add_argument("--pdb", required=True)
    pt.add_argument("--pdb_chain", default="A")
    pt.add_argument("--out_encoder", default="artefacts/public_ha_graph_encoder.pt")
    pt.add_argument("--graph_len", type=int, default=None)
    pt.add_argument("--max_pretrain_seqs", type=int, default=8000)
    pt.add_argument("--embed_dim", type=int, default=128)
    pt.add_argument("--encoder_layers", type=int, default=3)
    pt.add_argument("--dropout", type=float, default=0.1)
    pt.add_argument("--mask_prob", type=float, default=0.20)
    pt.add_argument("--pretrain_epochs", type=int, default=60)
    pt.add_argument("--pretrain_patience", type=int, default=10)
    pt.add_argument("--pretrain_batch_size", type=int, default=32)
    pt.add_argument("--pretrain_lr", type=float, default=1e-3)
    pt.add_argument("--distance_cutoff", type=float, default=8.0)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--cpu", action="store_true")

    # ---- train -----------------------------------------------------------
    tr = sub.add_parser("train", help="train + evaluate + save inference bundle")
    tr.add_argument("--data", required=True, help="collapsed MN CSV/XLSX")
    tr.add_argument("--virus_fasta", required=True)
    tr.add_argument("--serum_fasta", default=None,
                    help="only used when --serum_mode sequence")
    tr.add_argument("--serum_mode", choices=["index", "sequence"], default="index",
                    help="index: antisera are panel indices read from the titer "
                         "table; sequence: antisera are HA sequences from "
                         "--serum_fasta")
    tr.add_argument("--pdb", required=True)
    tr.add_argument("--pdb_chain", default="A")
    tr.add_argument("--pretrained_encoder", required=True)
    tr.add_argument("--gisaid_fasta", default=None,
                    help="public HA FASTA used to build the PSSM prior (recommended)")
    tr.add_argument("--output_dir", required=True)
    tr.add_argument("--graph_len", type=int, default=None)
    tr.add_argument("--max_pssm_seqs", type=int, default=11000)
    tr.add_argument("--distance_cutoff", type=float, default=8.0)

    tr.add_argument("--split_mode", default="stratified",
                    choices=["stratified", "ratio_holdout", "pair_holdout"])
    tr.add_argument("--test_size", type=float, default=0.10)
    tr.add_argument("--val_size", type=float, default=0.10)
    tr.add_argument("--use_replicates", action="store_true",
                    help="expand each training condition into its individual replicates")

    tr.add_argument("--n_seeds", type=int, default=5, help="ensemble size")
    tr.add_argument("--epochs", type=int, default=400)
    tr.add_argument("--patience", type=int, default=60)
    tr.add_argument("--batch_size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-3)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--dropout", type=float, default=0.15)
    tr.add_argument("--token_dim", type=int, default=128)
    tr.add_argument("--mix_hidden_dim", type=int, default=64)
    tr.add_argument("--lstm_hidden", type=int, default=128)
    tr.add_argument("--lstm_layers", type=int, default=2)
    tr.add_argument("--huber_delta", type=float, default=1.0)
    tr.add_argument("--label_noise_std", type=float, default=0.05)
    tr.add_argument("--weight_scheme", default="sqrt_inverse",
                    choices=["none", "inverse", "sqrt_inverse"])
    tr.add_argument("--max_titer_weight", type=float, default=5.0)
    tr.add_argument("--monitor", default="mae", choices=["mae", "rmse", "neg_r2", "neg_within1"])
    tr.add_argument("--tolerance", type=float, default=1.0)
    tr.add_argument("--bootstrap_n", type=int, default=1000)
    tr.add_argument("--ablation", default="full",
                    choices=["full", "no_mixture", "mixture_only", "sequence_only",
                             "no_sequence_embeddings", "no_pair_features"])
    tr.add_argument("--grid_step", type=float, default=0.1)
    tr.add_argument("--skip_grid", action="store_true")

    # ---- cross-validation protocols ----
    tr.add_argument("--cv", default="none",
                    choices=["none", "random_grouped", "leave_one_ratio",
                             "leave_one_pair", "all"],
                    help="additional CV evaluation run after the 80/10/10 fit")
    tr.add_argument("--cv_only", action="store_true",
                    help="run only the CV protocols; skip the final model, grid and bundle")
    tr.add_argument("--cv_n_seeds", type=int, default=3,
                    help="ensemble size within each CV fold (kept smaller than --n_seeds)")
    tr.add_argument("--cv_epochs", type=int, default=None,
                    help="epochs per CV fold (defaults to --epochs)")
    tr.add_argument("--cv_val_size", type=float, default=0.12,
                    help="fraction of each fold's training rows used for early stopping")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--cpu", action="store_true")
    tr.add_argument("--verbose", action="store_true")

    # ---- predict ---------------------------------------------------------
    pr = sub.add_parser("predict", help="inference for virus x serum x ratio")
    pr.add_argument("--bundle", required=True)
    pr.add_argument("--virus", default=None)
    pr.add_argument("--serum", default=None)
    pr.add_argument("--ratio", default="33:33:33", help="K:T:I, e.g. 40:35:25")
    pr.add_argument("--simplex", action="store_true", help="predict the whole K/T/I simplex")
    pr.add_argument("--grid_step", type=float, default=0.05)
    pr.add_argument("--query_csv", default=None,
                    help="CSV with virus_strain, serum_strain and either 'ratio' or K/T/I columns")
    pr.add_argument("--out_csv", default=None)
    pr.add_argument("--virus_fasta", default=None, help="needed only for strains not in the bundle")
    pr.add_argument("--serum_fasta", default=None)
    pr.add_argument("--cpu", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pretrain":
        pretrain_encoder(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "predict":
        run_prediction(args)


if __name__ == "__main__":
    main()
