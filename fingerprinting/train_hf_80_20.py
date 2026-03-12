import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI


DEFAULT_HF_REPO = "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH"


def build_data_config(args, model_path):
    return {
        "dataset_name": DatasetAPI.DATASET_AODT_HF,
        "samples_count": args.samples_count,
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
        "model_path": model_path,
    }


def build_model_config(args):
    model_config = {
        "batch_size": args.batch_size,
        "loss_type": args.loss_type,
        "alpha": args.alpha,
        "row": args.row,
        "enable_ind": args.enable_ind,
    }
    if args.loss_type == "quadruplet_loss":
        model_config["beta"] = args.beta
    return model_config


def run_training_and_test(args):
    root_dir = str(Path(__file__).resolve().parents[1])
    model_path = args.model_path or os.path.join(root_dir, "aodt_hf_models")
    os.makedirs(model_path, exist_ok=True)

    data_config = build_data_config(args, model_path=model_path)
    model_config = build_model_config(args)

    print("=== AODT HF Training Configuration ===")
    print(f"HF repo: {data_config['hf_repo_id']}")
    print(f"Split mode: train='{args.hf_train_split}', test='{args.hf_test_split}'")
    print(f"Requested train ratio: {args.train_ratio:.4f}")
    print(f"RX ID: {args.rx_id}")
    print(f"Model path: {model_path}")
    print(f"Model config: {model_config}")
    print(f"Samples count: {args.samples_count}")
    print()

    dataset_api = DatasetAPI(
        root_dir=root_dir,
        matlab_src_dir="",
        matlab_session_id="",
        aug_on=False,
    )
    extractor_api = ExtractorAPI()

    print("[1/2] Training feature extractor...")
    (
        data_train,
        labels_train,
        _,
        data_test,
        labels_test,
        _,
        node_ids_train,
        node_ids_test,
    ) = dataset_api.load_hf_train_test(data_config, shuffle_train=True, shuffle_test=False)

    data_train = data_train[:, 0 : data_config["samples_count"]]
    data_test = data_test[:, 0 : data_config["samples_count"]]

    shared_labels = sorted(list(set(node_ids_train).intersection(set(node_ids_test))))
    if not shared_labels:
        raise RuntimeError("No overlapping labels between train/test splits for closed-set evaluation.")

    train_mask = np.isin(labels_train.flatten(), shared_labels)
    test_mask = np.isin(labels_test.flatten(), shared_labels)
    data_train = data_train[train_mask]
    labels_train = labels_train[train_mask]
    data_test = data_test[test_mask]
    labels_test = labels_test[test_mask]

    model_file = os.path.join(model_path, f"extractor_{args.rx_id}.keras")
    feature_extractor, history_obj = extractor_api.train(
        data_train, labels_train, shared_labels, model_config, save_path=model_file
    )
    print("Training complete.")
    print(f"Saved model: {model_file}")
    if args.print_history:
        history = history_obj.history
        if "loss" in history and len(history["loss"]) > 0:
            print(f"Final train loss: {history['loss'][-1]:.6f}")
        if "val_loss" in history and len(history["val_loss"]) > 0:
            print(f"Final valid loss: {history['val_loss'][-1]:.6f}")

    if not args.skip_eval:
        print("\n[2/2] Evaluating on held-out test split...")
        fps_train = extractor_api.run(feature_extractor, data_train, model_config)
        fps_test = extractor_api.run(feature_extractor, data_test, model_config)
        knn = KNeighborsClassifier(n_neighbors=args.knn_k, metric="euclidean")
        knn.fit(fps_train, np.ravel(labels_train))
        labels_pred = knn.predict(fps_test)
        accuracy = accuracy_score(labels_test, labels_pred)
        print(f"Closed-set test accuracy (k={args.knn_k}): {accuracy:.4f}")
    else:
        print("\nEvaluation skipped (--skip-eval enabled).")

    return feature_extractor


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train HF fingerprint extractor on "
            "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH using 80/20 split."
        )
    )
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO, help="HF dataset repo id")
    parser.add_argument("--hf-revision", default=None, help="HF dataset revision/tag/branch")
    parser.add_argument("--hf-train-split", default="train", help="HF train split")
    parser.add_argument(
        "--hf-test-split",
        default="train",
        help="HF test split; keep equal to train to use ratio split",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default: 0.8)")
    parser.add_argument("--label-column", default="rnti", help="Label column in dataset")
    parser.add_argument("--iq-column", default="iq", help="IQ column in dataset")
    parser.add_argument("--rx-ant", type=int, default=0, help="RX antenna index")
    parser.add_argument(
        "--sym-mode",
        default="flatten",
        choices=["flatten", "first_sym", "mean_sym"],
        help="How IQ symbols are collapsed to 1D",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on train load")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap on test load")
    parser.add_argument("--samples-count", type=int, default=400, help="Number of IQ samples used")
    parser.add_argument("--model-path", default=None, help="Directory where models are saved")
    parser.add_argument("--rx-id", default=DatasetAPI.RX_1, help="Receiver/model id suffix")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--loss-type",
        default="triplet_loss",
        choices=["triplet_loss", "quadruplet_loss"],
        help="Metric learning loss type",
    )
    parser.add_argument("--alpha", type=float, default=1.1, help="Triplet/quadruplet alpha")
    parser.add_argument("--beta", type=float, default=0.37, help="Quadruplet beta")
    parser.add_argument("--row", type=int, default=80, help="STFT row/window size")
    parser.add_argument(
        "--enable-ind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable channel-independent transform",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation")
    parser.add_argument("--knn-k", type=int, default=10, help="K for closed-set KNN evaluation")
    parser.add_argument(
        "--print-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print final train/validation loss",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training_and_test(parse_args())
