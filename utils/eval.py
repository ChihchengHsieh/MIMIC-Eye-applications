import pickle, os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import List
from collections import OrderedDict
from .coco_eval import (
    CocoEvaluator,
    external_summarize,
    external_get_num_fps,
    external_get_num_fns,
    external_get_num_tps,
)
import utils.print as print_f
from models.load import  get_model_name, get_trained_model
from data.constants import DEFAULT_REFLACX_LABEL_COLS
# from utils.plot import plot_losses, plot_train_val_evaluators
from utils.train import num_params

# def get_ar_ap(
#     evaluator: CocoEvaluator,
#     areaRng: str = "all",
#     iouThr: float = 0.5,
#     maxDets: int = 10,
# ) -> Tuple[float, float]:

#     ar = external_summarize(
#         evaluator.coco_eval["bbox"],
#         ap=0,
#         areaRng=areaRng,
#         iouThr=iouThr,
#         maxDets=maxDets,
#         print_result=False,
#     )

#     ap = external_summarize(
#         evaluator.coco_eval["bbox"],
#         ap=1,
#         areaRng=areaRng,
#         iouThr=iouThr,
#         maxDets=maxDets,
#         print_result=False,
#     )

#     return ar, ap


def get_ap_ar(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    ap = external_summarize(
        evaluator.coco_eval["bbox"],
        ap=1,
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
        print_result=False,
    )

    ar = external_summarize(
        evaluator.coco_eval["bbox"],
        ap=0,
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
        print_result=False,
    )

    return {"ap": ap, "ar": ar}


def get_num_fps(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    num_fps = external_get_num_fps(
        evaluator.coco_eval["bbox"], iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    return num_fps


def get_num_fns(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    num_fns = external_get_num_fns(
        evaluator.coco_eval["bbox"], iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    return num_fns


def get_num_tps(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    num_tps = external_get_num_tps(
        evaluator.coco_eval["bbox"], iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    return num_tps


def get_ap_ar_for_train_val(
    train_evaluator: CocoEvaluator,
    val_evaluator: CocoEvaluator,
    iouThr=0.5,
    areaRng="all",
    maxDets=10,
):

    train_ap_ar = get_ap_ar(
        train_evaluator, iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    val_ap_ar = get_ap_ar(
        val_evaluator, iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    return train_ap_ar, val_ap_ar


def save_iou_results(evaluator: CocoEvaluator, suffix: str, model_path: str):
    ap_ar_dict = OrderedDict(
        {thrs: [] for thrs in evaluator.coco_eval["bbox"].params.iouThrs}
    )

    for thrs in evaluator.coco_eval["bbox"].params.iouThrs:
        test_ap_ar = get_ap_ar(evaluator, areaRng="all", maxDets=10, iouThr=thrs,)

        ap_ar_dict[thrs].append(test_ap_ar)

        print(
            f"IoBB [{thrs:.4f}] | AR [{test_ap_ar['ar']:.4f}] | AP [{test_ap_ar['ap']:.4f}]"
        )

    with open(
        os.path.join("eval_results", f"{model_path}_{suffix}.pkl"), "wb",
    ) as training_record_f:
        pickle.dump(ap_ar_dict, training_record_f)


def get_thrs_evaluation_df(
    models, dataset, disease="all", iobb_thrs=0.5, score_thrs=0.05
):
    all_models_eval_data = {}
    for select_model in models:
        with open(
            os.path.join(
                "eval_results",
                f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
            ),
            "rb",
        ) as f:
            eval_data = pickle.load(f)
            all_models_eval_data[select_model.value] = eval_data

    return pd.DataFrame(
        [
            {
                "model": str(select_model).split(".")[-1],
                **all_models_eval_data[select_model.value][iobb_thrs][0],
            }
            for select_model in models
        ]
    )[["model", "ap", "ar"]]


def plot_iou_result(
    models,
    datasets,
    naming_map,
    disease="all",
    figsize=(10, 10),
    include_recall=False,
    score_thrs=0.05,
):

    cm = plt.get_cmap("rainbow")
    NUM_COLORS = len(models)

    all_models_eval_data = {dataset: {} for dataset in datasets}

    for select_model in models:
        for dataset in datasets:
            with open(
                os.path.join(
                    "eval_results",
                    f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
                ),
                "rb",
            ) as f:
                eval_data = pickle.load(f)
                all_models_eval_data[dataset][select_model.value] = eval_data

    fig, axes = plt.subplots(
        len(datasets),
        2 if include_recall else 1,
        figsize=figsize,
        dpi=120,
        sharex=True,
        squeeze=False,
    )

    for i, dataset in enumerate(datasets):
        axes[i, 0].set_title(f"[{dataset}] - Average Precision")
        axes[i, 0].set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for select_model in models:
            axes[i, 0].plot(
                all_models_eval_data[dataset][select_model.value].keys(),
                [
                    v[0]["ap"]
                    for v in all_models_eval_data[dataset][select_model.value].values()
                ],
                marker="o",
                label=get_model_name(select_model, naming_map=naming_map),
                # color="darkorange",
            )
        axes[i, 0].legend(loc="lower left")
        axes[i, 0].set_xlabel("IoBB threshold")

        if include_recall:

            axes[i, 1].set_title(f"[{dataset}] - Average Recall")
            axes[i, 1].set_prop_cycle(
                "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
            )

            for select_model in models:
                axes[i, 1].plot(
                    all_models_eval_data[dataset][select_model.value].keys(),
                    [
                        v[0]["ar"]
                        for v in all_models_eval_data[dataset][
                            select_model.value
                        ].values()
                    ],
                    marker="o",
                    label=get_model_name(select_model, naming_map=naming_map),
                    # color="darkorange",
                )

            axes[i, 1].legend(loc="lower left")
            axes[i, 1].set_xlabel("IoBB threshold")

    plt.tight_layout()
    plt.plot()
    plt.pause(0.01)

    return fig


def showModelOnDatasets(
    select_model,
    datasets,
    naming_map,
    disease="all",
    figsize=(10, 10),
    include_recall=False,
    score_thrs=0.05,
):
    """
    This function used for detecting the overfitting dataset.    
    """
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(datasets)

    all_models_eval_data = {}
    for dataset in datasets:
        with open(
            os.path.join(
                "eval_results",
                f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
            ),
            "rb",
        ) as f:
            eval_data = pickle.load(f)
            all_models_eval_data[dataset] = eval_data

    fig, axes = plt.subplots(
        2 if include_recall else 1,
        figsize=figsize,
        dpi=120,
        sharex=True,
        squeeze=False,
    )

    axes = axes[0]

    fig.suptitle(get_model_name(select_model, naming_map=naming_map))

    axes[0].set_title("Average Precision")
    axes[0].set_prop_cycle(
        "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    )

    for dataset in datasets:
        axes[0].plot(
            all_models_eval_data[dataset].keys(),
            [v[0]["ap"] for v in all_models_eval_data[dataset].values()],
            marker="o",
            label=dataset,
            # color="darkorange",
        )
    axes[0].legend(loc="lower left")

    if include_recall:
        axes[1].set_title("Average Recall")
        axes[1].set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for dataset in datasets:
            axes[1].plot(
                all_models_eval_data[dataset].keys(),
                [v[0]["ar"] for v in all_models_eval_data[dataset].values()],
                marker="o",
                label=dataset,
                # color="darkorange",
            )

        axes[1].legend(loc="lower left")
        axes[1].set_xlabel("IoBB")

    plt.tight_layout()
    plt.plot()
    plt.pause(0.01)

    return fig

def showModelsOnDatasets(
    select_models, datasets, naming_map, disease="all", figsize=(10, 10),
    score_thrs=0.05,
):
    """
    This function used for detecting the overfitting dataset.    
    """
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(datasets)

    fig, axes = plt.subplots(
        1 , len(select_models) , figsize=figsize, dpi=120, sharex=True, sharey=True ,squeeze=False,
    )

    fig.suptitle("Average Precision")

    for c_i , select_model,in enumerate(select_models):
        all_models_eval_data = {}
        for dataset in datasets:
            with open(
                os.path.join(
                    "eval_results", f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
                ),
                "rb",
            ) as f:
                eval_data = pickle.load(f)
                all_models_eval_data[dataset] = eval_data


        ax = axes[0][c_i]

        ax.set_title(f"{get_model_name(select_model, naming_map=naming_map)}")
        ax.set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for dataset in datasets:
            ax.plot(
                all_models_eval_data[dataset].keys(),
                [v[0]["ap"] for v in all_models_eval_data[dataset].values()],
                marker="o",
                label=dataset,
                # color="darkorange",
            )
        ax.legend(loc="lower left")
        ax.set_xlabel("IoBB threshold")
    plt.tight_layout()
    plt.plot()
    plt.pause(0.01)

    return fig



def showModelOnScoreThrs(
    select_model,
    dataset: str,
    naming_map,
    disease="all",
    figsize=(10, 10),
    include_recall=False,
    score_thresholds=[0.5, 0.3, 0.2, 0.1, 0.05],
):
    """
    This function used for detecting the overfitting dataset.    
    """
    cm = plt.get_cmap("gist_rainbow")
    NUM_COLORS = len(score_thresholds)

    all_models_eval_data = {}
    for score_thrs in score_thresholds:
        with open(
            os.path.join(
                "eval_results",
                f"{select_model.value}_{dataset}_{disease}_score_thrs{score_thrs}.pkl",
            ),
            "rb",
        ) as f:
            eval_data = pickle.load(f)
            all_models_eval_data[score_thrs] = eval_data

    fig, axes = plt.subplots(
        2 if include_recall else 1, figsize=figsize, dpi=80, sharex=True, squeeze=False,
    )

    axes = axes[0]

    fig.suptitle(get_model_name(select_model, naming_map=naming_map))

    axes[0].set_title("Average Precision")
    axes[0].set_prop_cycle(
        "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    )

    for score_thrs in score_thresholds:
        axes[0].plot(
            all_models_eval_data[score_thrs].keys(),
            [v[0]["ap"] for v in all_models_eval_data[score_thrs].values()],
            marker="o",
            label=f"score_thrs={str(score_thrs)}",
            # color="darkorange",
        )
    axes[0].legend(loc="lower left")

    if include_recall:
        axes[1].set_title("Average Recall")
        axes[1].set_prop_cycle(
            "color", [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        )

        for score_thrs in score_thresholds:
            axes[1].plot(
                all_models_eval_data[score_thrs].keys(),
                [v[0]["ar"] for v in all_models_eval_data[score_thrs].values()],
                marker="o",
                label=f"score_thrs={str(score_thrs)}",
                # color="darkorange",
            )

        axes[1].legend(loc="lower left")
        axes[1].set_xlabel("IoBB")

    plt.plot()
    plt.pause(0.01)

    return fig

# def plot_training_progress(trained_models, device):
#     for trained_model in trained_models:
#         _, train_info, _ = get_trained_model(
#             trained_model,
#             DEFAULT_REFLACX_LABEL_COLS,
#             device,
#             rpn_nms_thresh=0.3,
#             box_detections_per_img=10,
#             box_nms_thresh=0.2,
#             rpn_score_thresh=0.0,
#             box_score_thresh=0.05,
#         )

#         print_f.print_title("Training Info")
#         print(train_info)

#         plot_train_val_evaluators(
#             train_ap_ars=train_info.train_ap_ars, val_ap_ars=train_info.val_ap_ars,
#         )

#         plot_losses(train_info.train_data, train_info.val_data)



# def print_num_params(trained_models, device):
#     for trained_model in trained_models:
#         model, train_info, _ = get_trained_model(
#             trained_model,
#             DEFAULT_REFLACX_LABEL_COLS,
#             device,
#             image_size=512,
#             rpn_nms_thresh=0.3,
#             box_detections_per_img=10,
#             box_nms_thresh=0.2,
#             rpn_score_thresh=0.0,
#             box_score_thresh=0.05,
#         )

#         print(f"| [{train_info.model_setup.name}] | #Params: [{num_params(model):,}] |")




def get_mAP_mAR(
    models,
    datasets: List[str],
    naming_map,
    score_thrs: float = 0.05,
):

    labels_cols = DEFAULT_REFLACX_LABEL_COLS + ["all"]
    # remove the labels that has "/" sign.
    labels_cols = [l.replace("/", "or") for l in labels_cols]
    # score_thrs = 0.05

    all_df = {d: {} for d in labels_cols}

    for disease_str in labels_cols:
        for select_model in models:
            model_path = select_model.value
            eval_df = pd.read_csv(
                os.path.join(
                    "eval_results",
                    f"{model_path}_{disease_str}_score_thrs{score_thrs}.csv",
                ),
                index_col=0,
            )
            all_df[disease_str][model_path] = eval_df

    # eval_dataset = 'val' # ['test', 'val', 'our']

    for eval_dataset in datasets:
        model_dfs = OrderedDict({})

        for select_model in models:
            model_path = select_model.value
            model_name = get_model_name(
                select_model, naming_map=naming_map
            )  # str(select_model).split(".")[-1]
            # Pick dataset

            model_eval = []
            for disease_str in labels_cols:
                model_eval.append(
                    {
                        **dict(
                            all_df[disease_str][model_path][
                                all_df[disease_str][model_path]["dataset"]
                                == eval_dataset
                            ].iloc[0]
                        ),
                        "disease": disease_str,
                    }
                )

            # model_dfs[model_name] = pd.DataFrame(model_eval)[
            #     ["disease", f"AP@[IoBB = 0.50:0.95]", f"AR@[IoBB = 0.50:0.95]"]
            # ]

            model_dfs[model_name] = pd.DataFrame(model_eval)[
                ["disease", f"AP@[IoBB = 0.50]", f"AR@[IoBB = 0.50]"]
            ]

        for idx, k in enumerate(model_dfs.keys()):
            if idx == 0:
                # create the merged df
                merged_df = model_dfs[k].copy()
                merged_df.columns = [
                    "disease" if c == "disease" else f"{c}_{k}"
                    for c in merged_df.columns
                ]
            else:
                df = model_dfs[k].copy()
                df.columns = [
                    "disease" if c == "disease" else f"{c}_{k}" for c in df.columns
                ]
                merged_df = merged_df.merge(df, "left", on="disease",)

        print_f.print_title(f"Dataset [{eval_dataset}]")
        display(merged_df)

        merged_df.to_csv(
            os.path.join(f"{eval_dataset}_dataset_class_ap_score_thrs_{score_thrs}.csv")
        )

        return merged_df