import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from tqdm.auto import tqdm

from utils.inference import DMMRLoss, read_images, normalize_data_


def get_scores(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path

    source_path = Path(args.source_path)
    target_path = Path(args.target_path)

    source_image, source_chns = read_images(source_path, names={'img'}, device=device)
    target_image, target_chns = read_images(target_path, names={'img'}, device=device)
    norm_params = {'scale': {'img': 1.0}, 'shift': {'img': 0.0}}

    source_image = normalize_data_(source_image, source_chns, **norm_params)
    target_image = normalize_data_(target_image, target_chns, **norm_params)

    subject = tio.Subject(
        source=tio.ScalarImage(tensor=source_image.cpu()),
        target=tio.ScalarImage(tensor=target_image.cpu()),
    )
    base_tfm = tio.ToCanonical()
    subject = base_tfm(subject)

    loss = DMMRLoss(model_path=model_path,
                    zero_percentage_threshold=args.zero_percentage_threshold,
                    patch_size=args.patch_size).to(device)

    angle_start, angle_stop, angle_step = map(int, args.angle_range.split(':'))
    angle_range = np.arange(angle_start, angle_stop, angle_step)

    rot_scores = []

    for angle in angle_range:
        source = subject['source'][tio.DATA].to(device)
        target = subject['target'][tio.DATA].unsqueeze(0).to(device)

        if args.axis == 'xyz':
            degrees = (angle, angle, angle, angle, angle, angle)
        elif args.axis == 'x':
            degrees = (angle, angle, 0, 0, 0, 0)
        elif args.axis == 'y':
            degrees = (0, 0, angle, angle, 0, 0)
        elif args.axis == 'z':
            degrees = (0, 0, 0, 0, angle, angle)
        else:
            raise ValueError(f'Unknown axis: {args.axis}')

        rot = tio.RandomAffine(scales=0, degrees=degrees)
        moving_rot = rot(source.cpu()).unsqueeze(0).to(device)
        score = loss(target=target, source=moving_rot)
        rot_scores.append(score.item())

    t_start, t_stop, t_step = map(int, args.translation_range.split(':'))
    t_range = np.arange(t_start, t_stop, t_step)

    t_scores = []

    for t in t_range:
        source = subject['source'][tio.DATA].to(device)
        target = subject['target'][tio.DATA].unsqueeze(0).to(device)

        if args.axis == 'xyz':
            translation = (t, t, t, t, t, t)
        elif args.axis == 'x':
            translation = (t, t, 0, 0, 0, 0)
        elif args.axis == 'y':
            translation = (0, 0, t, t, 0, 0)
        elif args.axis == 'z':
            translation = (0, 0, 0, 0, t, t)
        else:
            raise ValueError(f'Unknown axis: {args.axis}')

        translate = tio.RandomAffine(scales=0, degrees=0, translation=translation)
        moving_t = translate(source.cpu()).unsqueeze(0).to(device)
        score = loss(target=target, source=moving_t)
        t_scores.append(score.item())

    return rot_scores, t_scores


def create_range(range_str):
    start, stop, step = map(int, range_str.split(':'))
    return np.arange(start, stop, step)


def get_subject_pairs(imgs_folder):
    set_name = imgs_folder.split('/')[-1]
    subject_paths = list(Path(imgs_folder).glob('sub-*'))
    subject_paths = [str(p) for p in subject_paths]
    sorted_pairs = {tuple(sorted([p1, p2])) for p1 in subject_paths for p2 in subject_paths if p1 != p2}
    subject_pairs = list(sorted_pairs)
    if set_name == 'train' or set_name == 'test':  # reduce the number of pairs for training or testing set
        total_len_4 = len(subject_pairs) // 20
        subject_pairs = subject_pairs[:total_len_4]
    return subject_pairs


def calculate_scores_subjects(args, subject_pairs, subject_paths):
    rot_inter_scores = []
    t_inter_scores = []
    rot_intra_scores = []
    t_intra_scores = []

    for source, target in tqdm(subject_pairs, desc='Inter-subject'):
        args.source_path = Path(source, 'T1_brain.nii.gz')
        args.target_path = Path(target, 'T2_brain.nii.gz')
        r_score, t_score = get_scores(args)
        rot_inter_scores.append(r_score)
        t_inter_scores.append(t_score)

    for f in tqdm(subject_paths, desc='Intra-subject'):
        args.source_path = Path(f, 'T1_brain.nii.gz')
        args.target_path = Path(f, 'T2_brain.nii.gz')
        r_score, t_score = get_scores(args)
        rot_intra_scores.append(r_score)
        t_intra_scores.append(t_score)

    return rot_inter_scores, t_inter_scores, rot_intra_scores, t_intra_scores


def plot_similarity_curves(set_name, model, angle_range, t_range, rot_inter_scores, t_inter_scores, rot_intra_scores,
                           t_intra_scores):
    # Plot rotation similarity curves
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{set_name}_{model}\nRotation ({args.axis} axis)')

    if rot_inter_scores:
        axs[0, 0].boxplot(np.array(rot_inter_scores), labels=angle_range)
        axs[0, 0].set_xlabel("θ_rot" + "\u1D62")
        axs[0, 0].set_ylabel('DMMR Score')
        axs[0, 0].set_xticklabels(angle_range, rotation=45)
        axs[0, 0].set_title('Inter-subject')

        for i, score in enumerate(rot_inter_scores):
            axs[0, 1].plot(angle_range, score, label=f'Subject {i + 1}', alpha=0.5)
        axs[0, 1].set_xlabel("θ_rot" + "\u1D62")
        axs[0, 1].set_ylabel('DMMR Score')
        axs[0, 1].set_title('Inter-subject')
        axs[0, 1].set_xlim(angle_range[0], angle_range[-1])

    axs[1, 0].boxplot(np.array(rot_intra_scores), labels=angle_range)
    axs[1, 0].set_xlabel("θ_rot" + "\u1D62")
    axs[1, 0].set_ylabel('DMMR Score')
    axs[1, 0].set_xticklabels(angle_range, rotation=45)
    axs[1, 0].set_title('Intra-subject')

    for i, score in enumerate(rot_intra_scores):
        axs[1, 1].plot(angle_range, score, label=f'Subject {i + 1}', alpha=0.5)
    axs[1, 1].set_xlabel("θ_rot" + "\u1D62")
    axs[1, 1].set_ylabel('DMMR Score')
    axs[1, 1].set_title('Intra-subject')
    axs[1, 1].set_xlim(angle_range[0], angle_range[-1])

    plt.tight_layout()
    # if not rot_inter_scores:
    plt.savefig(f'outputs/figures/{set_name}_rot_trans_curves/rotation/'
                f'{set_name}_combined_{model}_rotation{args.angle_range}_axis{args.axis}_'
                f'{args.zero_percentage_threshold}zeropercent.png')
    plt.show()

    # Plot translation similarity curves
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{set_name}_{model.split(".pt")[0]}\nTranslation ({args.axis} axis)')

    if t_inter_scores:
        axs[0, 0].boxplot(np.array(t_inter_scores), labels=t_range)
        axs[0, 0].set_xlabel("θ" + "\u1D62")
        axs[0, 0].set_ylabel('DMMR Score')
        axs[0, 0].set_xticklabels(t_range, rotation=45)
        axs[0, 0].set_title('Inter-subject')

        for i, score in enumerate(t_inter_scores):
            axs[0, 1].plot(t_range, score, label=f'Subject {i + 1}', alpha=0.5)
        axs[0, 1].set_xlabel("θ" + "\u1D62")
        axs[0, 1].set_ylabel('DMMR Score')
        axs[0, 1].set_title('Inter-subject')
        axs[0, 1].set_xlim(t_range[0], t_range[-1])

    axs[1, 0].boxplot(np.array(t_intra_scores), labels=t_range)
    axs[1, 0].set_xlabel("θ" + "\u1D62")
    axs[1, 0].set_ylabel('DMMR Score')
    axs[1, 0].set_xticklabels(t_range, rotation=45)
    axs[1, 0].set_title('Intra-subject')

    for i, score in enumerate(t_intra_scores):
        axs[1, 1].plot(t_range, score, label=f'Subject {i + 1}', alpha=0.5)
    axs[1, 1].set_xlabel("θ" + "\u1D62")
    axs[1, 1].set_ylabel('DMMR Score')
    axs[1, 1].set_title('Intra-subject')
    axs[1, 1].set_xlim(t_range[0], t_range[-1])

    plt.tight_layout()
    # if not t_inter_scores:
    plt.savefig(f'outputs/figures/{set_name}_rot_trans_curves/translation/'
                f'{set_name}_combined_{model}_translation{args.translation_range}_axis{args.axis}_'
                f'{args.zero_percentage_threshold}zeropercent.png')
    plt.show()


def run_experiment(args):
    # Initialize the models list
    models = [
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs100_nonorm.pt',
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs100_norm.pt',
        # 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs79_online_aug.pt',
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs64_online_aug.pt',
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs50_intersubject.pt',
        # 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs50_intersubject.pt',
        # 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs69_online_aug_extra_tfms.pt',  # -> same as single axis rot
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs69_online_aug_extra_tfms.pt',
        # 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs100_online_aug_tuned_tfms.pt',  # -> no single axis rot,
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs59_online_aug_tuned_tfms.pt',  # just 3 axis rot at same time
        # 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs100_online_aug_tuned_tfms_single_axis.pt',
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs34_online_aug_tuned_tfms_single_axis.pt',
        # 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt',
        # 'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt',
        # 'camcan_t1t1_dmmr_net_sigmoid_bce_lr0.0001_epochs89_online_aug_tuned_tfms_single_axis_small_rot.pt',
        # 'camcan_t1t1_dmmr_net_tanh_hinge_lr0.0001_epochs89_online_aug_tuned_tfms_single_axis_small_rot.pt',
        # 'camcan_t1t1_dmmr_net_sigmoid_bce_lr0.0001_epochs19_online_aug_tuned_tfms_single_axis.pt',
        # 'camcan_t1t1_dmmr_net_tanh_hinge_lr0.0001_epochs59_online_aug_tuned_tfms_single_axis.pt',
        # 'dmmr_ixi_tanh.pt',
        # 'dmmr_ixi_sigmoid.pt'
        'camcan_t1t2_dmmr_net_tanh_hinge_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt',
        'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs59_online_aug_tuned_tfms_single_axis_small_rot_bound.pt',
        'camcan_t1t2_dmmr_net_tanh_hinge_ps34.pt',
        'camcan_t1t2_dmmr_net_sigmoid_bce_ps34.pt',
        'camcan_t1t2_dmmr_net_sigmoid_multiclass_ce_5pos_5neg.pt',
        'camcan_t1t2_dmmr_net_sigmoid_multiclass_ce_1pos_4neg.pt',
        'camcan_t1t2_dmmr_net_sigmoid_multiclass_ce_1pos_4neg_ps34.pt',
        'camcan_t1t2_dmmr_net_sigmoid_multiclass_ce_1pos_1neg.pt',
        'camcan_t1t2_dmmr_net_sigmoid_multiclass_ce_1pos_1neg_ps34.pt',
    ]

    for model in models:
        print(f'Running experiment for model {model}...')
        args.model_path = f'outputs/dmmr_models/{model}'

        angle_range = create_range(args.angle_range)
        t_range = create_range(args.translation_range)
        set = 'test'
        data_path = f'/home/joao/data/camcan_malpem/{set}'
        subject_paths = list(Path(data_path).glob('sub-*'))
        if set == 'train' or set == 'test':
            subject_paths = subject_paths[:len(subject_paths) // 5]
        subject_pairs = get_subject_pairs(data_path)[:len(subject_paths)]

        if 'ps34' in model:
            args.patch_size = 34

        (rot_inter_scores, t_inter_scores,
         rot_intra_scores, t_intra_scores) = calculate_scores_subjects(args,
                                                                       subject_pairs,
                                                                       subject_paths)

        plot_similarity_curves(set, model,
                               angle_range, t_range,
                               rot_inter_scores, t_inter_scores,
                               rot_intra_scores, t_intra_scores)

    return 0


def exp_zero_percentage(args):
    model = 'camcan_t1t2_dmmr_net_sigmoid_bce_lr0.0001_epochs54_online_aug_tuned_tfms_single_axis_small_rot.pt'
    print(f'Running experiment w/ zero percentage for model {model}...')
    args.model_path = f'outputs/dmmr_models/{model}'

    angle_range = create_range(args.angle_range)
    t_range = create_range(args.translation_range)
    data_path = '/home/joao/data/camcan_malpem/test'
    subject_pairs = get_subject_pairs(data_path)[:1]
    subject_paths = list(Path(data_path).glob('sub-*'))[:1]

    zero_percentage_range = np.round(np.arange(0, 1.1, 0.1), 2)
    for axis in ['x', 'y', 'z', 'xyz']:
        rot_inter_scores_list = []
        rot_intra_scores_list = []
        t_inter_scores_list = []
        t_intra_scores_list = []
        args.axis = axis
        for zero_percentage in zero_percentage_range:
            print(60*'-')
            print(f'Zero percentage {zero_percentage}...')
            args.zero_percentage_threshold = zero_percentage

            (rot_inter_scores, t_inter_scores,
             rot_intra_scores, t_intra_scores) = calculate_scores_subjects(args,
                                                                           subject_pairs,
                                                                           subject_paths)

            rot_inter_scores_list.append(rot_inter_scores[0])
            rot_intra_scores_list.append(rot_intra_scores[0])
            t_inter_scores_list.append(t_inter_scores[0])
            t_intra_scores_list.append(t_intra_scores[0])


        set_name = data_path.split('/')[-1]
        fig, axs = plt.subplots(1, 4, figsize=(25, 10))
        fig.suptitle(f'{set_name}_{model}\nR + T ({args.axis} axis) - Effect of zero percentage threshold')

        for rot_inter, rot_intra, t_inter, t_intra, zero_percent in zip(rot_inter_scores_list, rot_intra_scores_list,
                                                                        t_inter_scores_list, t_intra_scores_list,
                                                                        zero_percentage_range):
            alpha = 1.1 - zero_percent if zero_percent != 0 else 1
            axs[0].plot(angle_range, rot_inter,
                        alpha=alpha, label=f'{zero_percent:.1f}')
            axs[1].plot(angle_range, rot_intra,
                        alpha=alpha, label=f'{zero_percent:.1f}')
            axs[2].plot(t_range, t_inter,
                        alpha=alpha, label=f'{zero_percent:.1f}')
            axs[3].plot(t_range, t_intra,
                        alpha=alpha, label=f'{zero_percent:.1f}')

        # Add legends with corresponding zero_percentage values
        for ax in axs:
            ax.legend(zero_percentage_range)

        plt.tight_layout()
        plt.savefig(f'outputs/figures/_zero_percent/{model}_{set_name}_{args.axis}_zero_percentage.png')
        plt.show()

    return 0


def main(args):
    angle_range = create_range(args.angle_range)
    t_range = create_range(args.translation_range)

    rot_scores, t_scores = get_scores(args)

    plot_similarity_curves(args.source_path.split('/')[-1], args.model_path,
                           angle_range, t_range,
                           [rot_scores], [t_scores],
                           None, None)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to get similarity curves for a trained dmmr model.")
    parser.add_argument("--source_path", type=str, help="Path to the source image")
    parser.add_argument("--target_path", type=str, help="Path to the target image")
    parser.add_argument("--model_path", type=str, help="Path to the jit model")
    parser.add_argument("--angle_range", default="-180:180:10",
                        type=str, help="Range of angles in format 'start:stop:step'")
    parser.add_argument("--translation_range", default="-100:100:10",
                        type=str, help="Range of translation in format 'start:stop:step'")
    parser.add_argument("--zero_percentage_threshold", default=0.2, type=float,)
    parser.add_argument("--patch_size", default=17, type=int,)
    parser.add_argument("--axis", type=str, default='xyz', help="Axis to rotate/translate around")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    # exp_zero_percentage(args)
    # run_experiment(args)
    sys.exit(main(args))
