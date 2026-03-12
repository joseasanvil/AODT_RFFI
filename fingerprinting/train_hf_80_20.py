import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI


DEFAULT_HF_REPO = "CAAI-FAU/PHY-Device-Fingerprinting-cuPHY-PUSCH"


def build_data_config(args, model_path):
    return {
        "dataset_name": DatasetAPI.DATASET_AODT_HF,
        "samples_count": args.samples_count,
        "hf_required_iq_len": args.required_iq_len,
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


def _plot_label_distribution(labels_train, labels_test, labels_order, output_path):
    train_counts = [int(np.sum(labels_train == lbl)) for lbl in labels_order]
    test_counts = [int(np.sum(labels_test == lbl)) for lbl in labels_order]

    x = np.arange(len(labels_order))
    width = 0.38

    plt.figure(figsize=(10, 5), dpi=140)
    bars_train = plt.bar(x - width / 2, train_counts, width=width, label="Train", color="#1f77b4")
    bars_test = plt.bar(x + width / 2, test_counts, width=width, label="Test", color="#ff7f0e")
    plt.xticks(x, [str(lbl) for lbl in labels_order])
    plt.xlabel("Device ID (label)")
    plt.ylabel("Number of samples")
    plt.title("Train/Test Sample Distribution per Device ID")
    plt.legend()
    max_height = max(train_counts + test_counts) if (train_counts or test_counts) else 1
    y_offset = max(1, int(max_height * 0.01))
    for bar in list(bars_train) + list(bars_test):
        height = int(bar.get_height())
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            str(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_confusion(labels_true, labels_pred, labels_order, output_path):
    cm = confusion_matrix(labels_true, labels_pred, labels=labels_order)
    cm = cm.astype(np.int32)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=140)

    # Absolute confusion counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks(np.arange(len(labels_order)))
    axes[0].set_yticks(np.arange(len(labels_order)))
    axes[0].set_xticklabels([str(x) for x in labels_order], rotation=45, ha="right")
    axes[0].set_yticklabels([str(x) for x in labels_order])
    max_count = max(1, int(cm.max()))
    count_thresh = max_count / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            text_color = "white" if val > count_thresh else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Normalized confusion (row-wise)
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (Row-normalized)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_xticks(np.arange(len(labels_order)))
    axes[1].set_yticks(np.arange(len(labels_order)))
    axes[1].set_xticklabels([str(x) for x in labels_order], rotation=45, ha="right")
    axes[1].set_yticklabels([str(x) for x in labels_order])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            pct = float(cm_norm[i, j]) * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            axes[1].text(j, i, f"{pct:.1f}%", ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return cm


def _print_top_confusions(cm, labels_order, top_k=5):
    failures = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt > 0:
                failures.append((cnt, labels_order[i], labels_order[j]))

    failures.sort(reverse=True, key=lambda x: x[0])
    if not failures:
        print("No off-diagonal confusions found.")
        return

    print("Top confusion pairs (true -> predicted):")
    for cnt, y_true, y_pred in failures[:top_k]:
        print(f"  {y_true} -> {y_pred}: {cnt}")


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
    print(f"Required IQ length: {args.required_iq_len}")
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
    labels_train = labels_train[train_mask].flatten().astype(int)
    data_test = data_test[test_mask]
    labels_test = labels_test[test_mask].flatten().astype(int)

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
        knn.fit(fps_train, labels_train)
        labels_pred = knn.predict(fps_test)
        accuracy = accuracy_score(labels_test, labels_pred)
        print(f"Closed-set test accuracy (k={args.knn_k}): {accuracy:.4f}")

        if args.plot_outputs:
            plot_dir = args.plot_dir or os.path.join(model_path, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            labels_order = sorted(list(set(labels_train).union(set(labels_test))))
            dist_plot = os.path.join(plot_dir, f"label_distribution_{args.rx_id}.png")
            cm_plot = os.path.join(plot_dir, f"confusion_matrix_{args.rx_id}.png")

            _plot_label_distribution(labels_train, labels_test, labels_order, dist_plot)
            cm = _plot_confusion(labels_test, labels_pred, labels_order, cm_plot)
            _print_top_confusions(cm, labels_order, top_k=args.top_confusions)

            print(f"Saved distribution plot: {dist_plot}")
            print(f"Saved confusion matrix plot: {cm_plot}")
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
    parser.add_argument(
        "--required-iq-len",
        type=int,
        default=39168,
        help="Keep only records with this exact post-processed IQ length",
    )
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
        "--plot-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save confusion matrix and train/test distribution plots",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory for plot outputs (default: <model_path>/plots)",
    )
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=5,
        help="Number of top off-diagonal confusions to print",
    )
    parser.add_argument(
        "--print-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print final train/validation loss",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training_and_test(parse_args())
