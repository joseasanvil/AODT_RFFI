import argparse
from pathlib import Path

import numpy as np

from dataset_api import DatasetAPI


def _format_ratio(train_count, test_count):
    total = train_count + test_count
    if total == 0:
        return np.nan
    return train_count / total


def validate_data_loading(args):
    root_dir = str(Path(__file__).resolve().parents[1])
    dataset_api = DatasetAPI(
        root_dir=root_dir,
        matlab_src_dir="",
        matlab_session_id="",
        aug_on=False,
        seed=args.seed,
    )

    data_config = {
        "dataset_name": DatasetAPI.DATASET_AODT_HF,
        "hf_repo_id": args.hf_repo_id,
        "hf_revision": args.hf_revision,
        "hf_train_split": args.hf_train_split,
        "hf_test_split": args.hf_test_split,
        "hf_train_ratio": args.train_ratio,
        "hf_label_column": args.label_column,
        "hf_iq_column": args.iq_column,
        "hf_rx_ant": args.rx_ant,
        "hf_sym_mode": args.sym_mode,
        "hf_max_train_samples": args.max_train_samples,
        "hf_max_test_samples": args.max_test_samples,
    }

    (
        data_train,
        labels_train,
        _,
        data_test,
        labels_test,
        _,
        node_ids_train,
        node_ids_test,
    ) = dataset_api.load_hf_train_test(data_config, shuffle_train=False, shuffle_test=False)

    train_count = int(labels_train.shape[0])
    test_count = int(labels_test.shape[0])
    total_count = train_count + test_count
    ratio_observed = _format_ratio(train_count, test_count)

    print("=== HF Data Loading Validation ===")
    print(f"Repo: {args.hf_repo_id}")
    print(f"Split mode: train='{args.hf_train_split}', test='{args.hf_test_split}'")
    print(f"Requested train ratio: {args.train_ratio:.4f}")
    print(f"Observed counts: train={train_count}, test={test_count}, total={total_count}")
    print(f"Observed train ratio: {ratio_observed:.4f}")
    print(f"Train tensor shape: {data_train.shape}")
    print(f"Test tensor shape:  {data_test.shape}")
    print(f"Unique labels train: {len(node_ids_train)}")
    print(f"Unique labels test:  {len(node_ids_test)}")

    validation_errors = []

    if train_count == 0:
        validation_errors.append("No training samples loaded.")
    if test_count == 0:
        validation_errors.append("No test samples loaded.")
    if data_train.ndim != 2 or data_test.ndim != 2:
        validation_errors.append("Expected 2D complex IQ arrays for train/test.")
    if data_train.shape[1] != data_test.shape[1]:
        validation_errors.append("Train/test feature lengths do not match.")

    if args.hf_train_split == args.hf_test_split:
        tolerance = args.ratio_tolerance
        if not np.isnan(ratio_observed) and abs(ratio_observed - args.train_ratio) > tolerance:
            validation_errors.append(
                f"Observed ratio {ratio_observed:.4f} is outside tolerance ±{tolerance:.4f}."
            )

        print("\nPer-device split check:")
        all_labels = sorted(set(labels_train.flatten().astype(int)).union(set(labels_test.flatten().astype(int))))
        for dev in all_labels:
            dev_train = int(np.sum(labels_train.flatten() == dev))
            dev_test = int(np.sum(labels_test.flatten() == dev))
            dev_ratio = _format_ratio(dev_train, dev_test)
            print(
                f"  label={dev:>4}  train={dev_train:>5}  test={dev_test:>5}  "
                f"train_ratio={dev_ratio:.4f}"
            )

    if validation_errors:
        print("\n[FAIL] Data loading is not training-ready:")
        for err in validation_errors:
            print(f"  - {err}")
        return 1

    print("\n[PASS] Data loading is valid and training-ready.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate HF data loading and 80/20 train-test split readiness."
    )
    parser.add_argument("--hf-repo-id", required=True, help="HF dataset repo id, e.g. org/name")
    parser.add_argument("--hf-revision", default=None, help="HF dataset revision/branch/tag")
    parser.add_argument("--hf-train-split", default="train", help="HF train split name")
    parser.add_argument("--hf-test-split", default="train", help="HF test split name")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Expected train ratio")
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=0.03,
        help="Allowed absolute ratio deviation for same-split mode",
    )
    parser.add_argument("--label-column", default="rnti", help="Label column name")
    parser.add_argument("--iq-column", default="iq", help="IQ column name")
    parser.add_argument("--rx-ant", type=int, default=0, help="RX antenna index")
    parser.add_argument(
        "--sym-mode",
        default="flatten",
        choices=["flatten", "first_sym", "mean_sym"],
        help="IQ symbol aggregation mode",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train cap")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional test cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    exit_code = validate_data_loading(parse_args())
    raise SystemExit(exit_code)
