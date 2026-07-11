import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.sar_augment import SARAugMixAugmenter
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.sar_filter import sar_filter_and_loss
from utils.semantic_region import SemanticRegionLocator
from utils.text_anchors import load_text_anchor_file
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def safe_arch_name(arch):
    return arch.replace("/", "-").replace("@", "_").replace(" ", "")


def resolve_anchor_path(anchor_path, set_id, arch):
    if anchor_path is None:
        return None
    return anchor_path.format(set_id=set_id, dataset=set_id, arch=safe_arch_name(arch))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    sar_filter_result = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            elif args.sar_tpt and getattr(args, "current_anchor_payload", None) is not None:
                output, image_features = model.inference_with_features(inputs)
            else:
                output = model(inputs) 

            if args.sar_tpt and getattr(args, "current_anchor_payload", None) is not None and not args.disable_sar_filter:
                anchors = args.current_anchor_payload["anchors"].to(output.device)
                if selected_idx is not None:
                    output = output[selected_idx]
                    image_features = image_features[selected_idx]
                else:
                    loss, sar_filter_result = sar_filter_and_loss(
                        logits=output,
                        image_features=image_features,
                        anchors=anchors,
                        target_idx=None,
                        logit_scale=model.logit_scale.exp(),
                        lambda_anchor=args.lambda_anchor,
                        entropy_scale=args.entropy_scale,
                        reliable_ratio=args.reliable_ratio,
                        reliable_top_k=args.reliable_top_k,
                        min_reliable_views=args.min_reliable_views,
                        disable_anchor_filter=args.disable_anchor_filter,
                        disable_entropy_filter=args.disable_entropy_filter,
                    )
                    selected_idx = sar_filter_result.reliable_idx
                    output = sar_filter_result.reliable_logits
                if selected_idx is not None and j > 0:
                    loss = avg_entropy(output)
            else:
                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, args.selection_p)
                loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    if sar_filter_result is not None:
        args.last_sar_filter_debug = sar_filter_result.as_debug_dict()
    return


def main():
    args = parser.parse_args()
    if args.sar_tpt:
        if not args.tpt:
            print("=> --sar_tpt implies --tpt; enabling test-time prompt tuning.")
            args.tpt = True
        if args.cocoop:
            raise NotImplementedError("SAR-TPT currently supports the CoOp/TPT branch, not CoCoOp.")
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            if args.sar_tpt:
                anchor_path = resolve_anchor_path(args.anchor_path, set_id, args.arch)
                if anchor_path is None:
                    raise ValueError("--sar_tpt requires --anchor_path for stage-three guided augmentation")
                anchor_payload = load_text_anchor_file(anchor_path, map_location='cpu')
                if list(anchor_payload["classnames"]) != list(classnames):
                    raise ValueError(
                        "Anchor classnames do not match current evaluation class order for {}. "
                        "Rebuild anchors with the same dataset id/class order.".format(set_id)
                    )
                args.current_anchor_payload = anchor_payload
                semantic_locator = SemanticRegionLocator(
                    anchor_payload,
                    mask_top_ratio=args.mask_top_ratio,
                    min_mask_area_ratio=args.min_mask_area_ratio,
                    fallback=args.mask_fallback,
                    allow_non_vit_fallback=args.allow_non_vit_fallback,
                )
                data_transform = SARAugMixAugmenter(
                    base_transform,
                    preprocess,
                    n_views=args.batch_size-1,
                    augmix=len(set_id)>1,
                    severity=args.augmix_severity,
                    semantic_locator=semantic_locator,
                    model=model,
                    device=torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu"),
                    tau_cov=args.tau_cov,
                    max_crop_trials=args.max_crop_trials,
                    crop_scale=(args.crop_scale_min, args.crop_scale_max),
                    crop_ratio=(args.crop_ratio_min, args.crop_ratio_max),
                    hflip_p=args.hflip_p,
                    output_size=args.resolution,
                    bbox_padding_ratio=args.bbox_padding_ratio,
                )
                print("=> Using SAR region-guided augmentation with anchors: {}".format(anchor_path))
            else:
                args.current_anchor_payload = None
                data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1,
                                                augmix=len(set_id)>1)
            batchsize = 1
        else:
            args.current_anchor_payload = None
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        workers = 0 if args.sar_tpt else args.workers
        if args.sar_tpt and args.workers != 0:
            print("=> SAR augmentation owns the model during transform; overriding workers to 0.")
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=workers, pin_memory=True)
            
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        if args.sar_tpt and hasattr(data_transform, "stats"):
            print("=> SAR augmentation stats [{}]: {}".format(set_id, data_transform.stats.as_dict()))
            if getattr(args, "last_sar_filter_debug", None) is not None:
                print("=> SAR filter last debug [{}]: {}".format(set_id, args.last_sar_filter_debug))
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, args)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(image)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sar_tpt', action='store_true', default=False,
                        help='use SAR-TPT region-guided multi-view augmentation')
    parser.add_argument('--anchor_path', default=None, type=str,
                        help='path to stage-one text anchors; supports {set_id}, {dataset}, {arch} placeholders')
    parser.add_argument('--mask_top_ratio', default=0.3, type=float,
                        help='stage-two semantic mask top activation ratio')
    parser.add_argument('--min_mask_area_ratio', default=0.01, type=float,
                        help='minimum semantic mask area before fallback')
    parser.add_argument('--mask_fallback', default='center', choices=['center', 'full', 'none'],
                        help='semantic mask fallback policy')
    parser.add_argument('--allow_non_vit_fallback', action='store_true', default=False,
                        help='allow center/full fallback mask when the visual backbone is not ViT')
    parser.add_argument('--tau_cov', default=0.6, type=float,
                        help='minimum semantic mask coverage required for guided crops')
    parser.add_argument('--max_crop_trials', default=20, type=int,
                        help='maximum candidate crop resampling attempts per view')
    parser.add_argument('--crop_scale_min', default=0.5, type=float,
                        help='minimum RandomResizedCrop scale for SAR guided views')
    parser.add_argument('--crop_scale_max', default=1.0, type=float,
                        help='maximum RandomResizedCrop scale for SAR guided views')
    parser.add_argument('--crop_ratio_min', default=3.0/4.0, type=float,
                        help='minimum RandomResizedCrop aspect ratio for SAR guided views')
    parser.add_argument('--crop_ratio_max', default=4.0/3.0, type=float,
                        help='maximum RandomResizedCrop aspect ratio for SAR guided views')
    parser.add_argument('--hflip_p', default=0.5, type=float,
                        help='horizontal flip probability for SAR guided views')
    parser.add_argument('--bbox_padding_ratio', default=0.15, type=float,
                        help='padding ratio for semantic-mask bounding-box fallback')
    parser.add_argument('--augmix_severity', default=1, type=int,
                        help='AugMix severity for SAR guided views')
    parser.add_argument('--lambda_anchor', default=0.5, type=float,
                        help='stage-four score weight for anchor similarity')
    parser.add_argument('--entropy_scale', default=1.0, type=float,
                        help='stage-four entropy score scale')
    parser.add_argument('--reliable_ratio', default=0.5, type=float,
                        help='stage-four ratio of reliable views to keep')
    parser.add_argument('--reliable_top_k', default=None, type=int,
                        help='stage-four fixed number of reliable views to keep; overrides reliable_ratio when > 0')
    parser.add_argument('--min_reliable_views', default=1, type=int,
                        help='minimum reliable views kept by SAR filter')
    parser.add_argument('--disable_sar_filter', action='store_true', default=False,
                        help='ablation: use SAR guided crops but original entropy filtering')
    parser.add_argument('--disable_anchor_filter', action='store_true', default=False,
                        help='ablation: remove anchor similarity from SAR filtering')
    parser.add_argument('--disable_entropy_filter', action='store_true', default=False,
                        help='ablation: remove prediction entropy from SAR filtering')

    main()
