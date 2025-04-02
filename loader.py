import pickle
import copy

import polars as pl
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from preprocessing import prepare_data_for_torchmetrics


def load_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


data = load_data("all_data_processed.pkl")


def compute_metrics(data_dict, **kwargs):
    """Compute mAP and mAR metrics using TorchMetrics"""
    preds, targets = prepare_data_for_torchmetrics(data_dict)

    metric = MeanAveragePrecision(**kwargs)
    metric.update(preds, targets)

    results = metric.compute()

    # Create a cleaner summary
    summary = {
        "mAP_50": results["map_50"].item(),
        "mAR_100": results["mar_100"].item(),
    }

    return summary, results


def compute_metrics_for_active_objects(data_dict, **kwargs):
    """Compute metrics only for active objects"""
    # Create a deep copy to avoid modifying the original data
    data_dict = copy.deepcopy(data_dict)
    active_data = {}

    for key in data_dict:
        active_data[key] = {}
        active_data[key]["ground_truth"] = {}
        active_data[key]["detic_data"] = {}

        # Filter ground truth to only active objects
        gt_boxes = []
        gt_classes = []
        active_flags = data_dict[key]["ground_truth"]["active"]

        for i, is_active in enumerate(active_flags):
            if is_active:
                gt_boxes.append(data_dict[key]["ground_truth"]["boxes"][i])
                gt_classes.append(data_dict[key]["ground_truth"]["classes"][i])

        # Update ground truth with only active objects
        if gt_boxes:
            active_data[key]["ground_truth"]["boxes"] = np.array(gt_boxes)
            active_data[key]["ground_truth"]["classes"] = gt_classes
        else:
            # Handle case with no active objects
            active_data[key]["ground_truth"]["boxes"] = np.zeros((0, 4))
            active_data[key]["ground_truth"]["classes"] = []

        # Filter detections to only active objects
        det_boxes = []
        det_scores = []
        det_classes = []
        det_active_flags = data_dict[key]["detic_data"]["active"]

        for i, is_active in enumerate(det_active_flags):
            if is_active:
                det_boxes.append(data_dict[key]["detic_data"]["boxes"][i])
                det_scores.append(data_dict[key]["detic_data"]["scores"][i])
                det_classes.append(data_dict[key]["detic_data"]["classes"][i])

        # Update detections with only active objects
        if det_boxes:
            active_data[key]["detic_data"]["boxes"] = np.array(det_boxes)
            active_data[key]["detic_data"]["scores"] = det_scores
            active_data[key]["detic_data"]["classes"] = det_classes
        else:
            # Handle case with no active detections
            active_data[key]["detic_data"]["boxes"] = np.zeros((0, 4))
            active_data[key]["detic_data"]["scores"] = []
            active_data[key]["detic_data"]["classes"] = []

    return compute_metrics(active_data, **kwargs)


if __name__ == "__main__":

    full_results, _ = compute_metrics(data)
    active_results, _ = compute_metrics_for_active_objects(data)

    full_micro_results, _ = compute_metrics(data, average="micro")
    active_micro_results, _ = compute_metrics_for_active_objects(data, average="micro")

    full_df = pl.concat(
        [
            pl.DataFrame(full_micro_results),
            pl.DataFrame(full_results),
            pl.DataFrame(active_micro_results),
            pl.DataFrame(active_results),
        ]
    ).with_columns(
        pl.Series(
            "Subset", ["full(micro)", "full(macro)", "active(micro)", "active(macro)"]
        )
    )

    print(full_df)
