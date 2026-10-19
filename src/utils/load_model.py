import logging
import os
import torch
from copy import deepcopy


# def load_roma(args, test_orginal_megadepth=False):
#     import sys
#     sys.path.append("./third_party/RoMa/")
#     from third_party.RoMa.romatch import roma_outdoor
#     from third_party.RoMa.romatch import tiny_roma_v1_outdoor
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_dense import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_roma import DataIOWrapper, lower_config
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
#     if args.ckpt2 == 'large':
#         # print('loading large roma model')
#         if args.ckpt is not None:
#             # print('loading roma model with weights')
#             pth_path = args.ckpt
#             state_dict = torch.load(pth_path, map_location=device)
#             matcher = roma_outdoor(device=device, weights=state_dict)
#         else:
#             matcher = roma_outdoor(device=device)
#     else:
#         # print('loading tiny roma model')
#         matcher = tiny_roma_v1_outdoor(device=device)
#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher


# def load_loftr(args, test_orginal_megadepth=False):
#     from third_party.LoFTR.src.loftr import LoFTR, default_cfg
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_dense import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_loftr import DataIOWrapper, lower_config
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     # print("default_cfg['coarse']['temp_bug_fix']", default_cfg['coarse']['temp_bug_fix'])
#     _default_cfg = deepcopy(default_cfg)
#     filename = os.path.basename(args.ckpt)
#     if filename != "outdoor_ds.ckpt":
#         # print('not using official old model, now change bug')
#         _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt

#     _default_cfg['match_coarse']['thr'] = args.thr
#     # print('now using thr:', args.thr)
#     matcher = LoFTR(config=_default_cfg)
#     matcher.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=True)
#     matcher = matcher.eval()

#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher

# def load_eloftr(args, test_orginal_megadepth=False):
#     from third_party.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_dense import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_loftr import DataIOWrapper, lower_config
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     # print("default_cfg['coarse']['temp_bug_fix']", default_cfg['coarse']['temp_bug_fix'])
#     _default_cfg = deepcopy(full_default_cfg)
#     filename = os.path.basename(args.ckpt)
#     if filename != "outdoor_ds.ckpt":
#         # print('not using official old model, now change bug')
#         _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt

#     _default_cfg['match_coarse']['thr'] = args.thr
#     # print('now using thr:', args.thr)
#     matcher = LoFTR(config=_default_cfg)
#     matcher.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=True)
#     matcher = matcher.eval()

#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher

# def load_sp_lg(args, test_orginal_megadepth=False):
#     from third_party.LightGlue.lightglue import LightGlue, SuperPoint
#     from third_party.LightGlue.lightglue.utils import rbd
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_sparse import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_sp_lg import DataIOWrapper, lower_config

#     class Matching(torch.nn.Module):
#         def __init__(self, sp_conf, lg_conf):
#             super().__init__()
#             device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
#             self.extractor = SuperPoint(**sp_conf).eval().to(device)  # load the feature extractor
#             self.matcher = LightGlue(features='superpoint', **lg_conf).eval().to(device)  # load the matcher
#             n_layers = lg_conf['n_layers']
#             # print(f"n_layers: {n_layers}")
#             ckpt_path = args.ckpt
#             # rename old state dict entries
#             state_dict = torch.load(ckpt_path, map_location=device)
#             for i in range(n_layers):
#                 pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
#                 state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
#                 pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
#                 state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
#             self.matcher.load_state_dict(state_dict, strict=False)

#         def forward(self, batch):
#             # batch = {'image0': image0, 'image1': image1}
#             image0 = batch['image0']
#             image1 = batch['image1']
#             # extract local features
#             if test_orginal_megadepth:
#                 feats0 = self.extractor.extract(image0, resize=None)  # auto-resize the image, disable with resize=None
#                 feats1 = self.extractor.extract(image1, resize=None)
#             else:
#                 feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
#                 feats1 = self.extractor.extract(image1)

#             # match the features
#             matches01 = self.matcher({'image0': feats0, 'image1': feats1})
#             feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
#             matches = matches01['matches']  # indices with shape (K,2)
#             points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
#             points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
#             matching_scores0 = matches01['matching_scores0']
#             matching_scores = matching_scores0[matches[..., 0]]

#             return {'matching_scores': matching_scores, 'keypoints0': points0, 'keypoints1': points1}

#     sp_conf = {
#         "descriptor_dim": 256,
#         "nms_radius": 4,
#         "max_num_keypoints": 2048,
#         "detection_threshold": 0.0005,
#         "remove_borders": 4,
#     }
#     lg_conf = {
#         "name": "lightglue",  # just for interfacing
#         "input_dim": 256,  # input descriptor dimension (autoselected from weights)
#         "descriptor_dim": 256,
#         "add_scale_ori": False,
#         "n_layers": 9,
#         "num_heads": 4,
#         "flash": True,  # enable FlashAttention if available.
#         "mp": False,  # enable mixed precision
#         "depth_confidence": 0.95,  # early stopping, disable with -1
#         "width_confidence": 0.99,  # point pruning, disable with -1
#         "filter_threshold": 0.1,  # match threshold
#         "weights": None,
#     }
#     matcher = Matching(sp_conf, lg_conf)
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher


def load_xoftr(args, test_orginal_megadepth=False):
    # from third_party.XoFTR.src.xoftr import XoFTR
    from src.xoftr import XoFTR_Light_convnextv2, XoFTR_Light, XoFTR_dino, XoFTR_dino_resnet, XoFTR_dino_resnet_panmatch, XoFTR_dino_resnet_panmatch_light, XoFTR_dino_resnet_panmatch_light_withfpn
    from src.xoftr import XoFTR_dino_resnet_panmatch_without_fpn, GeoSAM_without_RopE, GeoSAM, GeoSAM_reverse_GGSA, \
        GeoSAM_ablation_structure
    from src.config.default import get_cfg_defaults
    # from src.config.default import get_cfg_defaults
    from src.utils.data_io import DataIOWrapper, lower_config
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    config["xoftr"]["match_coarse"]["thr"] = args.match_threshold
    config["xoftr"]["fine"]["thr"] = args.fine_threshold
    ckpt = args.ckpt
    # matcher = XoFTR_dino_resnet_panmatch_light_withfpn(config=config["xoftr"])
    matcher = XoFTR_dino_resnet_panmatch_light_withfpn(config=config["xoftr"])
    # matcher = XoFTR_Light(config=config["xoftr"])
    matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)
    logging.info(config["test"])
    return matcher

# def load_xfeat(args, test_orginal_megadepth=False):
#     import imageio as imio
#     import numpy as np
#     import tqdm
#     from third_party.Xfeat.modules.xfeat import XFeat
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_sparse import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_xfeat import DataIOWrapper, lower_config

#     class Matching(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

#             self.matcher = XFeat(weights=args.ckpt).eval().to(device)

#         def forward(self, batch):
#             x = torch.randn(1, 3, 480, 640)
#             output = self.matcher.detectAndCompute(x, top_k=4096)[0]

#             # x = torch.randn(1, 3, 480, 640)
#             # # Stress test
#             # for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
#             #     output = self.matcher.detectAndCompute(x, top_k=4096)

#             image0 = batch['image0']
#             image1 = batch['image1']
#             mkpts_0, mkpts_1 = self.matcher.match_xfeat(image0, image1)

#             return {'keypoints0': mkpts_0, 'keypoints1': mkpts_1}

#     matcher = Matching()
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher

# def load_reasoning_xfeat(args, test_orginal_megadepth=False):
#     from reasoning.features.desc_reasoning import load_reasoning_from_checkpoint, Reasoning
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_sparse import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_reasoning_xfeat import DataIOWrapper, lower_config

#     class Matching(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
#             if not os.path.exists("models/xfeat.zip"):
#                 zip_weights = "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat.zip"
#                 os.makedirs("models", exist_ok=True)
#                 torch.hub.download_url_to_file(zip_weights, dst="models/xfeat.zip", progress=True)

#             # unzip
#             if not os.path.exists("models/xfeat/"):
#                 import zipfile
#                 with zipfile.ZipFile("models/xfeat.zip", 'r') as zip_ref:
#                     zip_ref.extractall("models/")

#             # Load the model with just the reasoning part
#             semantic_reasoning = load_reasoning_from_checkpoint('models/xfeat/')
#             self.matcher = Reasoning(semantic_reasoning['model']).eval().to(device)

#         def forward(self, batch):
#             # device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
#             image0 = batch['image0']
#             image1 = batch['image1']
#             image0 = image0.repeat(1, 3, 1, 1)  # 在通道维度重复3次
#             image1 = image1.repeat(1, 3, 1, 1)
#             match_response = self.matcher.match({
#                 'image0': image0,
#                 'image1': image1
#             })

#             mkpts0 = match_response['matches0'][0].cpu().numpy()
#             mkpts1 = match_response['matches1'][0].cpu().numpy()

#             return {'keypoints0': mkpts0, 'keypoints1': mkpts1}

#     matcher = Matching()
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher

# def load_omniglue(args, test_orginal_megadepth=False):
#     import third_party.Omniglue.src.omniglue
#     import cv2
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_sparse import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_omniglue import DataIOWrapper, lower_config

#     class Matching(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
#             self.og = third_party.Omniglue.src.omniglue.OmniGlue(
#                 og_export="./weights/omniglue_models/og_export",
#                 sp_export="./weights/omniglue_models/sp_v6",
#                 dino_export="./weights/omniglue_models/dinov2_vitb14_pretrain.pth",
#             )

#         def forward(self, batch):
#             # batch = {'image0': image0, 'image1': image1}
#             image0 = batch['image0']
#             image1 = batch['image1']
#             image0 = image0.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             image1 = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
#             image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
#             match_kp0, match_kp1, match_confidences = self.og.FindMatches(image0, image1)
#             match_threshold = 0.0  # Choose any value [0.0, 1.0).
#             keep_idx = []
#             for i in range(match_kp0.shape[0]):
#                 if match_confidences[i] > match_threshold:
#                     keep_idx.append(i)
#             match_kp0 = match_kp0[keep_idx]
#             match_kp1 = match_kp1[keep_idx]
#             match_confidences = match_confidences[keep_idx]

#             return {'matching_scores': match_confidences, 'keypoints0': match_kp0, 'keypoints1': match_kp1}

#     matcher = Matching()
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher

# def load_semaglue(args, test_orginal_megadepth=False):
#     from models.matching import Matching
#     import cv2
#     if test_orginal_megadepth:
#         from src.config.default_for_megadepth_sparse import get_cfg_defaults
#     else:
#         from src.config.default import get_cfg_defaults
#     from src.utils.data_io_semaglue import DataIOWrapper, lower_config

#     class Matching_semaglue(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

#             config = {
#                 'superpoint': {
#                     'nms_radius': 3,
#                     'keypoint_threshold': 0.005,
#                     'max_keypoints': 2048
#                 },
#             }
#             self.matcher = Matching(config).eval().to(device)

#         def forward(self, batch):
#             device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
#             image0 = batch['image0']
#             image1 = batch['image1']
#             pred = self.matcher({'image0': image0, 'image1': image1})
#             kpts0 = pred['keypoints0'][0].cpu().numpy()
#             kpts1 = pred['keypoints1'][0].cpu().numpy()
#             matches = pred['matches0'][0].cpu().numpy()
#             confidence = pred['matching_scores0'][0].cpu().numpy()

#             valid = matches > -1
#             mkpts0 = kpts0[valid]
#             mkpts1 = kpts1[matches[valid]]

#             return {'matching_scores': confidence, 'keypoints0': mkpts0, 'keypoints1': mkpts1}

#     matcher = Matching_semaglue()
#     config = get_cfg_defaults(inference=True)
#     config = lower_config(config)
#     matcher = DataIOWrapper(matcher, config=config["test"])
#     logging.info(config["test"])
#     return matcher

def load_model(method, args, use_path=True, test_orginal_megadepth=False):
    if use_path:
        matcher = eval(f"load_{method}")(args, test_orginal_megadepth=test_orginal_megadepth)
        return matcher.from_paths
    else:
        matcher = eval(f"load_{method}")(args, test_orginal_megadepth=test_orginal_megadepth)
        return matcher.from_cv_imgs
