import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
import transformers
from typing import Optional, List, Tuple
from PIL import Image
from dataclasses import dataclass, field
from transformers import SiglipImageProcessor
import torch.distributed as distributed
import zipfile
from enum import Enum

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import preprocess_image
from segearth_r2.utils.constants import IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX
from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset, EarthReasonDataset
import torch.nn.functional as F

# ==========================================
# Constants & Configuration
# ==========================================
DEFAULT_PREFIX_INST = (
    "This is an image \n<image>\n, please doing Reasoning Segmentation according to the following instruction:"
)

BASE_COLORS = [
    [255, 0, 0],   
    [0, 255, 0],    
    [0, 0, 255],    
    [255, 255, 0],  
    [255, 0, 255],  
    [0, 255, 255],  
    [255, 165, 0],  
    [128, 0, 128],  
]

@dataclass
class Arguments:
    local_rank: int = 0

    vision_tower: str = "pretrained_model/CLIP"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"

    lazy_preprocess: bool = False
    base_data_path: Optional[str] = field(default='your_data_path')
    
    model_path: Optional[str] = field(default="SegEarthR2_LaSeRS/hfweights-50000")
    mask_config: Optional[str] = field(default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'segearth_r2'
    version: str = 'llava_phi'
    
    temperature: float = 0.2
    num_beams: int = 1
    max_new_tokens: int = 128
    do_sample: bool = True

    output_dir: str = 'save_folder'
    dataloader_num_workers: int = 8
    dataset_type: str = field(default='LaSeRS')
    data_split: str = field(default='test')
    skip_empty_target: bool = False

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
 
 
class AverageMeter(object):
    """Computes and stores the average and current value."""
 
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
 
 
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """
    'K' classes, output and target sizes are N or N*L or N*H*W, values in [0, K-1].
    Returns per-class (area_intersection, area_union, area_target), each shape [K].
    """
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
 
 
def update_metrics_for_pair(pred_bin, gt_bin, device, intersection_meter, union_meter,
                             acc_iou_meter, pr_meters, thresholds):
    """
    pred_bin, gt_bin: single-image binary masks, any shape torch/numpy can broadcast
                       to [H, W], values in {0, 1}.
    Updates the running meters in place with this pair's per-class (bg/fg)
    intersection/union and thresholded IoU, exactly mirroring compute_metric()
    in the SegEarth-R1 eval script.
    """
    pred = torch.as_tensor(pred_bin).long().to(device).contiguous().view(-1)
    gt = torch.as_tensor(gt_bin).long().to(device).contiguous().view(-1)
 
    intersection, union, _ = intersectionAndUnionGPU(pred, gt, 2, ignore_index=255)
    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
 
    acc_iou = intersection / (union + 1e-5)
    acc_iou[union == 0] = 1.0
    foreground_iou = acc_iou[1]
 
    intersection_meter.update(intersection)
    union_meter.update(union)
    acc_iou_meter.update(acc_iou, n=1)
 
    for t in thresholds:
        pr_meters[t].update(1.0 if foreground_iou > t else 0.0, n=1)

# ==========================================
# Helper Functions
# ==========================================
def tokenizer_special_tokens(prompt: str, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, 
                             refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
    """Tokenize the prompt while preserving special multimodal tokens."""
    input_ids = []
    special_token_map = {'<image>': image_token_index, '<refer>': refer_token_index}
    prompt_chunks = re.split('(<image>|<refer>)', prompt)

    for chunk in prompt_chunks:
        if chunk in special_token_map:
            input_ids.append(special_token_map[chunk])
        elif chunk != '':
            input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
            
    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long).squeeze()
    elif return_tensors is not None:
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
        
    return input_ids

def preprocess_image_clip(image_path: str, clip_image_processor) -> torch.Tensor:
    """Read and preprocess the image specifically for the CLIP vision encoder."""
    img_clip = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_clip = clip_image_processor.preprocess(img_clip, return_tensors="pt")["pixel_values"][0]
    return image_clip

def preprocess_instruction(text: str, prefix_inst: str, tokenizer, conversation_lib) -> torch.Tensor:
    """Format the text instruction into the model's expected conversation template."""
    sources = [[{'from': 'human', 'value': prefix_inst + '\n' + text}, {'from': 'gpt', 'value': ''}]]

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:] # Skip the first one if it is not from human

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"Role mismatch at index {i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack(
        [tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
    )
    return input_ids[0]

def preprocess_input(text: str, image_path: str, tokenizer, clip_image_processor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orchestrate the preprocessing of text and image inputs."""
    instruction = text.strip()

    # Image preprocessing for main model
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    image_RGB = preprocess_image(image_path, 1024)
    image_tensor = torch.as_tensor(np.ascontiguousarray(image_RGB)).float()
    if image_tensor.shape[0] != 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    images = (image_tensor - pixel_mean) / pixel_std

    # Text and CLIP image preprocessing
    input_ids = preprocess_instruction(instruction, DEFAULT_PREFIX_INST, tokenizer, conversation_lib)
    images_clip = preprocess_image_clip(image_path, clip_image_processor)

    return input_ids.unsqueeze(0), images.unsqueeze(0), images_clip.unsqueeze(0)

def init_distributed_mode(para):
    para.distributed = True
    if torch.cuda.device_count() <= 1:
        para.distributed = False
        para.local_rank = 0
        para.world_size = 1

    if para.distributed:
         # Init distributed environment
        distributed.init_process_group(backend="nccl")

        local_rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        print('I am rank %d in this world of size %d!' % (local_rank, world_size))
        para.local_rank = local_rank
        para.world_size = world_size

def zip_folder(folder_path):
    folder_path = os.path.abspath(folder_path)
    folder_name = os.path.basename(folder_path)
    zip_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname=arcname)


def main():
    parser = transformers.HfArgumentParser(Arguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    model_path = os.path.expanduser(data_args.model_path)

    print("---------- Initializing Model ----------")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_args=data_args, 
        mask_config=data_args.mask_config, 
        device="cuda"
    )
    
    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu") 
    model.to(dtype=torch.float32, device=device)
    # model.eval() # Ensure model is in evaluation mode

    print("---------- Model Initialization Complete ----------")

    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version] # phi-2
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)

    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
    if data_args.dataset_type == 'LaSeRS':
        json_folders = os.path.join(data_args.base_data_path, 'rs_reason_seg/LaSeRS/test/annotations')
        splits = os.listdir(json_folders)
    elif data_args.dataset_type == 'EarthReason':
        splits = [data_args.data_split]
    else:
        raise ValueError(f"Unknown dataset_type: {data_args.dataset_type!r} (expected 'LaSeRS' or 'EarthReason')")
    # save_folder = data_args.output_dir

    for split in splits:
        print(f'------ cur benchmark is {data_args.dataset_type} [{split}] subset -------')
 
        if data_args.dataset_type == 'LaSeRS':
            eval_dataset = LaSeRSDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer,
                                          data_args=data_args, split=split)
        else:  # 'EarthReason'
            eval_dataset = EarthReasonDataset(base_data_path=data_args.base_data_path, tokenizer=tokenizer,
                                               data_args=data_args, split=split)
 
        dataloader_params = {
            "batch_size": 1,
            "num_workers": data_args.dataloader_num_workers,
        }
 
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=dataloader_params['batch_size'],
            shuffle=False,
            num_workers=dataloader_params['num_workers'],
            pin_memory=False,
            sampler=None,
            collate_fn=data_collator)
 
        do_eval(model, tokenizer, clip_image_processor, eval_dataloader, split, data_args, device)

def do_eval(model, tokenizer, clip_image_processor, eval_dataloader, split, data_args, device):
    model.eval()
 
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    pr_meters = {t: AverageMeter(f"Pr@{t}", ":6.3f", Summary.AVERAGE) for t in thresholds}
 
    with torch.no_grad():
        SEG_token_id = tokenizer.encode('[SEG]', add_special_tokens=False)[0]
        tokenizer.pad_token = tokenizer.eos_token
 
        overall_mask_num = 0
        n_skipped_empty = 0
 
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            mask_num = inputs['mask_num'][0]
 
            if mask_num == 0 or len(inputs['seg_info']) == 0:
                if data_args.skip_empty_target or len(inputs['seg_info']) == 0:
                    n_skipped_empty += 1
                    continue
 
            overall_mask_num += mask_num
 
            text = inputs['seg_info'][0].get('instruction', None) if 'instruction' in inputs['seg_info'][0] else inputs.get('ref', [None])[0]
            image_path = inputs['seg_info'][0]['image_path']
 
            input_ids, images, images_clip = preprocess_input(text, image_path, tokenizer, clip_image_processor)
            output_ids, masks_pred = model.inference(
                input_ids=input_ids.to(device),
                images=images.to(device),
                images_clip=images_clip.to(device),
                do_sample=data_args.do_sample,
                eos_token_id=tokenizer.eos_token_id,
                temperature=data_args.temperature,
                num_beams=data_args.num_beams,
                max_new_tokens=data_args.max_new_tokens,
                use_cache=True,
                SEG_token_id=SEG_token_id
            )
 
            gt_masks = []
            for _seg_info in inputs['seg_info']:
                gt_mask = _seg_info['mask'].unsqueeze(0).float()  # cast: bilinear needs float, not uint8
                gt_mask = F.interpolate(
                    gt_mask,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                gt_mask = (gt_mask > 0).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]
                gt_masks.append(gt_mask)
 
            n_gt = len(gt_masks)
            if n_gt == 0:
                # empty-target sample with skip_empty_target=False: score
                # against an all-zero GT mask at the model's input resolution.
                H, W = images.shape[-2], images.shape[-1]
                gt_masks = [np.zeros((H, W), dtype=np.uint8)]
                n_gt = 1
 
            if masks_pred is None:
                H, W = images.shape[-2], images.shape[-1]
                masks_pred = np.zeros((n_gt, 1, H, W), dtype=np.uint8)
            n_pred = masks_pred.shape[0]
 
            if n_pred < n_gt:
                masks_pred = np.concatenate(
                    [masks_pred, np.repeat(masks_pred[-1:], n_gt - n_pred, axis=0)], axis=0
                )
            elif n_pred > n_gt:
                masks_pred = masks_pred[:n_gt]
 
            for pred_mask, gt_mask in zip(masks_pred, gt_masks):
                pred_bin = (pred_mask.squeeze() > 0).astype(np.uint8)
                update_metrics_for_pair(pred_bin, gt_mask, device,
                                         intersection_meter, union_meter,
                                         acc_iou_meter, pr_meters, thresholds)
 
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]
 
        print(f"Overall mask num: {overall_mask_num}")
        if n_skipped_empty:
            print(f"Skipped (empty-target) samples: {n_skipped_empty}")
        print(f"{split} ciou: {ciou:.4f}, giou: {giou:.4f}")
        print("IoU Thresholds: " + ", ".join([f"@{t}: {m.avg:.4f}" for t, m in pr_meters.items()]))
 
 
if __name__ == "__main__":
    main()