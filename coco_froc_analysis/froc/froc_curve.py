from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..utils import build_gt_id2annotations
from ..utils import build_pr_id2annotations
from ..utils import COLORS
from ..utils import load_json_from_file
from ..utils import transform_gt_into_pr
from ..utils import update_scores
from .froc_stats import init_stats
from .froc_stats import update_stats
import itertools
from scipy import interpolate


def froc_point(gt, pr, score_thres, use_iou, iou_thres):
    
    pr = update_scores(pr, score_thres)

    categories = gt['categories']

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)

    stats = update_stats(
        stats, gt_id_to_annotation, pr_id_to_annotation,
        categories, use_iou, iou_thres,
    )

    return stats

# def froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres):
#     gt = load_json_from_file(gt_ann)
#     pr = load_json_from_file(pr_ann)

#     pr = update_scores(pr, score_thres)

#     categories = gt['categories']

#     stats = init_stats(gt, categories)

#     gt_id_to_annotation = build_gt_id2annotations(gt)
#     pr_id_to_annotation = build_pr_id2annotations(pr)

#     stats = update_stats(
#         stats, gt_id_to_annotation, pr_id_to_annotation,
#         categories, use_iou, iou_thres,
#     )

#     return stats


def calc_scores(stats, lls_accuracy, nlls_per_image):
    for category_id in stats:
        if lls_accuracy.get(category_id, None):
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] /
                stats[category_id]['n_lesions'],
            )
        else:
            lls_accuracy[category_id] = []
            lls_accuracy[category_id].append(
                stats[category_id]['LL'] /
                stats[category_id]['n_lesions'],
            )

        if nlls_per_image.get(category_id, None):
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )
        else:
            nlls_per_image[category_id] = []
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )

    return lls_accuracy, nlls_per_image


# def generate_froc_curve(
#     gt_ann,
#     pr_ann,
#     use_iou=False,
#     iou_thres=0.5,
#     n_sample_points=50,
#     plot_title='FROC curve',
#     plot_output_path='froc.png',
#     test_ann=None,
#     bounds=None,
# ):

#     lls_accuracy = {}
#     nlls_per_image = {}

#     for score_thres in tqdm(
#             np.linspace(0.0, 1.0, n_sample_points, endpoint=False),
#     ):
#         stats = froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres)
#         lls_accuracy, nlls_per_image = calc_scores(
#             stats, lls_accuracy,
#             nlls_per_image,
#         )

#     if plot_title:
#         fig, ax = plt.subplots(figsize=[27, 10])
#         ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
#         ins.set_xticks(
#             [0.1, 1.0, 2.0, 3.0, 4.0], [
#                 0.1, 1.0, 2.0, 3.0, 4.0,
#             ], fontsize=30,
#         )

#         if bounds is not None:
#             _, x_max, _, y_max = bounds
#             ins.set_xlim([.1, x_max])
#         else:
#             ins.set_xlim([0.1, 4.5])

#     for category_id in lls_accuracy:
#         lls = lls_accuracy[category_id]
#         nlls = nlls_per_image[category_id]
#         if plot_title:
#             ax.semilogx(
#                 nlls,
#                 lls,
#                 'x--',
#                 label='AI ' + stats[category_id]['name'],
#             )
#             ins.plot(
#                 nlls,
#                 lls,
#                 'x--',
#                 label='AI ' + stats[category_id]['name'],
#             )

#             if test_ann is not None:
#                 for t_ann, c in zip(test_ann, COLORS):
#                     t_ann, label = t_ann
#                     t_pr = transform_gt_into_pr(t_ann, gt_ann)
#                     stats = froc_point(gt_ann, t_pr, .5, use_iou, iou_thres)
#                     _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
#                     if plot_title:
#                         ax.plot(
#                             _nlls_per_image[category_id][0],
#                             _lls_accuracy[category_id][0],
#                             'D',
#                             markersize=15,
#                             markeredgewidth=3,
#                             label=label +
#                             f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
#                             c=c,
#                         )
#                         ins.plot(
#                             _nlls_per_image[category_id][0],
#                             _lls_accuracy[category_id][0],
#                             'D',
#                             markersize=12,
#                             markeredgewidth=2,
#                             label=label +
#                             f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
#                             c=c,
#                         )
#                         ax.hlines(
#                             y=_lls_accuracy[category_id][0],
#                             xmin=np.min(nlls),
#                             xmax=np.max(nlls),
#                             linestyles='dashed',
#                             colors=c,
#                         )
#                         ins.hlines(
#                             y=_lls_accuracy[category_id][0],
#                             xmin=np.min(nlls),
#                             xmax=np.max(nlls),
#                             linestyles='dashed',
#                             colors=c,
#                         )
#                         ax.text(
#                             x=np.min(nlls), y=_lls_accuracy[category_id][0] - 0.02,
#                             s=f' FP/image = {np.round(_nlls_per_image[category_id][0], 2)}',
#                             fontdict={'fontsize': 20, 'fontweight': 'bold'},
#                         )

#     if plot_title:
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

#         ax.legend(
#             loc='center left', bbox_to_anchor=(.8, .6),
#             fancybox=True, shadow=True, ncol=1, fontsize=25,
#         )

#         ax.set_title(plot_title, fontdict={'fontsize': 35})
#         ax.set_ylabel('Sensitivity', fontdict={'fontsize': 30})
#         ax.set_xlabel('FP / image', fontdict={'fontsize': 30})

#         ax.tick_params(axis='both', which='major', labelsize=30)
#         ins.tick_params(axis='both', which='major', labelsize=20)

#         if bounds is not None:
#             x_min, x_max, y_min, y_max = bounds
#             ax.set_ylim([y_min, y_max])
#             ax.set_xlim([x_min, x_max])
#         else:
#             ax.set_ylim(bottom=0.05, top=1.02)
#         fig.tight_layout(pad=2.0)
#         fig.savefig(fname=plot_output_path, dpi=150)
#     else:
#         return lls_accuracy, nlls_per_image


def generate_froc_curve(
    gt,
    pr,
    use_iou=True,
    iou_thres=0.5,
    n_sample_points=50,
    plot_title='FROC curve',
    plot_output_path='froc.png',
    bounds=None,
):
    lls_accuracy = {}
    nlls_per_image = {}

    for score_thres in tqdm(
            np.linspace(0.0, 1.0, n_sample_points, endpoint=False),
    ):
        stats = froc_point(gt, pr, score_thres, use_iou, iou_thres)
        # stats[-1]['n_images'] = stats[-1]['n_images'] * len(gt['categories']) # because it go through the image multiple times.
        lls_accuracy, nlls_per_image = calc_scores(
            stats, lls_accuracy,
            nlls_per_image,
        )

    if plot_title:
        fig, ax = plt.subplots(figsize=[27, 10])
        ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
        ins.set_xticks(
            [0.1, 1.0, 2.0, 3.0, 4.0], [
                0.1, 1.0, 2.0, 3.0, 4.0,
            ], fontsize=30,
        )

        if bounds is not None:
            _, x_max, _, y_max = bounds
            ins.set_xlim([.1, x_max])
        else:
            ins.set_xlim([0.1, 4.5])

    for category_id in lls_accuracy:
        lls = lls_accuracy[category_id]
        nlls = nlls_per_image[category_id]
        if plot_title:
            ax.semilogx(
                nlls,
                lls,
                'x--',
                label=stats[category_id]['name'],
            )
            ins.plot(
                nlls,
                lls,
                'x--',
                label=stats[category_id]['name'],
            )
           
    if plot_title:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(
            loc='center left', bbox_to_anchor=(.8, .6),
            fancybox=True, shadow=True, ncol=1, fontsize=25,
        )

        ax.set_title(plot_title, fontdict={'fontsize': 35})
        ax.set_ylabel('Sensitivity', fontdict={'fontsize': 30})
        ax.set_xlabel('FP / image', fontdict={'fontsize': 30})

        ax.tick_params(axis='both', which='major', labelsize=30)
        ins.tick_params(axis='both', which='major', labelsize=20)

        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            ax.set_ylim([y_min, y_max])
            ax.set_xlim([x_min, x_max])
        else:
            ax.set_ylim(bottom=0.05, top=1.02)
        fig.tight_layout(pad=2.0)
        fig.savefig(fname=plot_output_path, dpi=150)
        return stats, lls_accuracy, nlls_per_image
    else:
        return stats, lls_accuracy, nlls_per_image
    


def get_froc_curve(
    dataset,
    dts,
    all_gts,
    plot_title=None,
    use_iou=True,
    n_sample_points=100,
    froc_save_folder="./froc_figures",
):
    anns = []
    gts = [
        g["lesion-detection"]
        for g in list(itertools.chain.from_iterable(list(all_gts)))
    ]
    for g in gts:
        for i, bb in enumerate(g["unsized_boxes"]):
            anns.append(
                {
                    "image_id": g["image_id"].item(),
                    "category_id": g["labels"][i].item(),
                    "iscrowd": g["iscrowd"][i].item(),
                    "bbox": bb.tolist(),
                    "ignore": 0,
                }
            )

    pr = []
    for dt in dts:
        for image_id, d in dt.items():
            for i in range(len(d["boxes"])):
                pr.append(
                    {
                        "image_id": image_id,
                        "category_id": d["labels"][i].item(),
                        "score": d["scores"][i].item(),
                        "bbox": d["boxes"][i].numpy().tolist(),
                    }
                )

    categories = [
        {
            "id": dataset.disease_to_idx(i),
            "name": i,
            "color": "blue",
        }
        for i in dataset.labels_cols
    ]

    gt = {
        "categories": categories,
        "annotations": anns,
        "images": [{"id": id} for id in list(set([g["image_id"].item() for g in gts]))],
    }

    if plot_title:
        os.makedirs(froc_save_folder, exist_ok=True)

    stats, lls_accuracy, nlls_per_image = generate_froc_curve(
        gt=gt,
        pr=pr,
        use_iou=use_iou,
        iou_thres=0.5,
        n_sample_points=n_sample_points,
        plot_title=plot_title,
        plot_output_path=os.path.join(froc_save_folder, f"{plot_title}.png")
        if plot_title
        else None,
        bounds=None,
    )

    return stats, lls_accuracy, nlls_per_image



def get_interpolate_froc(
    stats, lls_accuracy,nlls_per_image, cat_id=None, fps_per_img=[0.5, 1, 2, 4], weight=True,
):
    
    if cat_id is None:
        number_all_lesions =sum([stats[cat_id]['n_lesions'] for cat_id in stats.keys()])
        all_frocs = []
        for cat_id in stats.keys():
            f = interpolate.interp1d(
                nlls_per_image[cat_id], lls_accuracy[cat_id], fill_value="extrapolate"
            )

            cat_froc = f(fps_per_img)


            if weight:
                cat_froc = cat_froc * ( stats[cat_id]["n_lesions"] /number_all_lesions) 

            all_frocs.append(cat_froc)

        if weight:
            return np.array(all_frocs).sum(axis=0)
        
        else:
            return np.array(all_frocs).mean(axis=0)

    else:
        f = interpolate.interp1d(
            nlls_per_image[cat_id], lls_accuracy[cat_id], fill_value="extrapolate"
        )
        return f(fps_per_img)
