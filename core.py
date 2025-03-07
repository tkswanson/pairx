import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from zennit.composites import EpsilonPlus
from xai_canonization.quantus_evaluation.canonizers.efficientnet import EfficientNetBNCanonizer
import torchvision.transforms as transforms
from zennit.torchvision import ResNetCanonizer
from custom_canonizers import ResNetCanonizerTimm
import timm
import torchvision
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import rbd
import pickle


COLORS = ['Grey', 'Purple', 'Blue', 'Green', 'Orange', 'Red']
CMAPS = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']


def get_intermediate_feature_maps_and_embedding(img, model, layer_keys):
    intermediate_fms = {}
    def get_intermediate_hook(name):
        def intermediate_hook(module, input, output):
            intermediate_fms[name] = output
            
        return intermediate_hook

    handles = []
    for layer_key in layer_keys:
        submodule = model.get_submodule(layer_key)
        assert submodule, f'could not find layer with key {layer_key}'
        
        handles.append(submodule.register_forward_hook(get_intermediate_hook(layer_key)))
        
    embedding = model(img)
    
    for handle in handles:    
        handle.remove()

    return intermediate_fms, embedding

def get_feature_matches(feature_map_0, feature_map_1, img_0, img_1, match_img_save_path=None):
    def flatten_to_descriptors(feature_map):
        descriptors = feature_map.squeeze().flatten(start_dim=1).transpose(0,1)
        return descriptors.detach().cpu().numpy()

    def get_keypoints(feature_map, img):
        h, w = feature_map.shape[-2:]
        img_h, img_w = img.shape[-2:]
        step_w = float(img_w) / float(w)
        step_h = float(img_h) / float(h)
    
        keypoints = [cv2.KeyPoint(x = step_w*(i+0.5),
                                  y = step_h*(j+0.5),
                                  size=1) for j in range(h) for i in range(w)]

        return keypoints

    def idx_to_coord(idx, feature_map):
        return (idx % feature_map.shape[-1], idx // feature_map.shape[-1])

    descriptors_0 = flatten_to_descriptors(feature_map_0)
    descriptors_1 = flatten_to_descriptors(feature_map_1)
    keypoints_0 = get_keypoints(feature_map_0, img_0)
    keypoints_1 = get_keypoints(feature_map_1, img_1)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_0, descriptors_1)

    if match_img_save_path:
        print("saving image with matches!")
        matched_image = cv2.drawMatches(
            to_displayable_np(img_0),
            keypoints_0,
            to_displayable_np(img_1),
            keypoints_1,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite(match_img_save_path, matched_image)
        print("saved!")

    matches = [{'coord0': idx_to_coord(match.queryIdx, feature_map_0),
                'coord1': idx_to_coord(match.trainIdx, feature_map_1),
                'keypoint0': keypoints_0[match.queryIdx],
                'keypoint1': keypoints_1[match.trainIdx],
                'distance': match.distance} for match in matches]

    return matches

def choose_canonizer(model):
    if type(model) is torchvision.models.resnet.ResNet:
        canonizer = ResNetCanonizer()
        #canonizer = EfficientNetBNCanonizer()
    elif (type(model) is timm.models.efficientnet.EfficientNet
          or (hasattr(model, 'backbone')
              and type(model.backbone) is timm.models.efficientnet.EfficientNet)):
        canonizer = EfficientNetBNCanonizer()
    elif isinstance(model, timm.models.resnet.ResNet):
        canonizer = ResNetCanonizerTimm()
    else:
        raise Exception("Model type not recognized for canonizer selection. Try explicitly passing a canonizer in.")

    return canonizer

def get_intermediate_relevances(img, gradient, model, layer_keys):
    composite = EpsilonPlus(canonizers=[choose_canonizer(model)])

    img.requires_grad = True
    img.grad = None

    with composite.context(model) as modified_model:
        intermediate_relevances = {}
        def save_grad(name):
            def hook(module, grad_in, grad_out):
                intermediate_relevances[name] = grad_out[0].squeeze().sum(dim=0).abs().detach().cpu().numpy()

            return hook

        handles = []
        for layer_key in layer_keys:
            submodule = model.get_submodule(layer_key)
            assert submodule, f'could not find layer with key {layer_key}'
            handles.append(submodule.register_full_backward_hook(save_grad(layer_key)))
        
        output = modified_model(img)
        output.backward(gradient=gradient)

        for handle in handles:
            handle.remove()
        
    return intermediate_relevances

def get_pixel_relevance(device, img, coord, model, layer_key):
    composite = EpsilonPlus(canonizers=[choose_canonizer(model)])
    
    img.requires_grad = True
    img.grad = None

    with composite.context(model) as modified_model:
        # TODO - does this have to have both a forward *and* backward pass every time?
        intermediate_fms, _ = get_intermediate_feature_maps_and_embedding(img, modified_model, [layer_key])
        intermediate_fm = intermediate_fms[layer_key]

        gradient = torch.zeros(intermediate_fm.shape)
        gradient[:, :, coord[1], coord[0]] = intermediate_fm[:, :, coord[1], coord[0]]

        intermediate_fm.backward(gradient=gradient.to(device))

    return img.grad.squeeze().sum(dim=0).abs().detach().cpu().numpy()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def inverse_normalize(image):
    tfm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / x for x in STD]),
            transforms.Normalize(mean=[-x for x in MEAN], std=[1.0, 1.0, 1.0]),
        ]
    )

    return tfm(image)

def to_displayable_np(img):
    return inverse_normalize(img.to('cpu'))[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def display_image_with_heatmap(img, heatmap, min = None, max = None):
    if min == None:
        min = np.min(heatmap)
    if max == None:
        max = np.max(heatmap)

    heatmap_scaled = 255 * (heatmap - min) / (max - min)
    heatmap_resized = cv2.resize(heatmap_scaled, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap_rgb, 0.4, 0)
    return overlay

def draw_matches(img_0, img_1, matches):
    output_img = cv2.hconcat((img_0, img_1))
    left_width = img_0.shape[1]
    
    for i, match in enumerate(matches):
        kp_0 = match['keypoint0'].pt
        kp_1 = match['keypoint1'].pt
        
        coord_0 = [int(kp_0[0]), int(kp_0[1])]
        coord_1 = [int(kp_1[0]) + left_width, int(kp_1[1])]
        
        color = tuple(int(c * 255) for c in mcolors.to_rgb(COLORS[i % len(COLORS)]))

        cv2.line(output_img, coord_0, coord_1, color, 1)
        cv2.circle(output_img, coord_0, 3, color, 2)
        cv2.circle(output_img, coord_1, 3, color, 2)

    return output_img

def draw_color_maps(value_set_0, value_set_1, img_shape):
    def map_color(values, cmap, gamma):
        mapper = cm.ScalarMappable(norm=mcolors.PowerNorm(gamma=gamma, vmin=0, vmax=np.max(values)), cmap=cmap)
        return 1 - mapper.to_rgba(values)[..., :3]

    assert len(value_set_0) == len(value_set_1), "value sets for color maps should have the same lengths"

    #output_img = np.zeros((value_set_0[0].shape[0], value_set_0[0].shape[1] + value_set_1[0].shape[1], 3))
    output_img = np.zeros(img_shape)
    if len(value_set_0) == 0:
        return output_img.astype(np.uint8)
    
    #scale_factor = min(max((4 / len(value_set_0)), 0.2), 1)
    #gamma = 1 - .02 * len(value_set_0)
    #gamma = 0.8 if len(value_set_0) <= 12 else 0.3 # TODO - figure out better scaling method - this is kind of model-dependent

    scale_factor = .8
    gamma = .95

    for i, (values_0, values_1) in enumerate(zip(value_set_0, value_set_1)):
        cmap = CMAPS[i % len(CMAPS)]
        color_mapped_0 = map_color(values_0, cmap, gamma)
        color_mapped_1 = map_color(values_1, cmap, gamma)

        output_img += cv2.hconcat([color_mapped_0, color_mapped_1])*scale_factor

    return (np.clip(1 - output_img, 0, 1) * 255).astype(np.uint8)

def draw_matches_and_color_maps(img_0, img_1, matches,
                                intermediate_relevance_0, intermediate_relevance_1,
                                pixel_relevances_0, pixel_relevances_1):
    img_np_0 = to_displayable_np(img_0)
    img_np_1 = to_displayable_np(img_1)

    img_hm_0 = display_image_with_heatmap(img_np_0, intermediate_relevance_0)
    img_hm_1 = display_image_with_heatmap(img_np_1, intermediate_relevance_1)

    matches_img = draw_matches(img_hm_0, img_hm_1, matches)
    color_map_img = draw_color_maps(pixel_relevances_0, pixel_relevances_1, matches_img.shape)

    return cv2.vconcat((matches_img, color_map_img))

def get_lightglue_homography(device, img_0, img_1):
    if 'TORCH_HOME' in os.environ:
        del os.environ['TORCH_HOME']

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    feats_0 = extractor.extract(img_0)
    feats_1 = extractor.extract(img_1)
    matches = matcher({"image0": feats_0, "image1": feats_1})
    
    feats_0, feats_1, matches = [rbd(x) for x in (feats_0, feats_1, matches)]
    kpts_0, kpts_1, matches = feats_0["keypoints"], feats_1["keypoints"], matches["matches"]

    m_kpts_0, m_kpts_1 = kpts_0[matches[..., 0]].cpu().numpy(), kpts_1[matches[..., 1]].cpu().numpy()

    try:
        H_lightglue, mask = cv2.findHomography(m_kpts_0, m_kpts_1, method=0)
    except:
        H_lightglue = None

    return H_lightglue

def calculate_residuals(H, src_pts, dst_pts):
    # TODO - just double check the math here
    src_pts = src_pts.reshape(-1, 2)
    dst_pts = dst_pts.reshape(-1, 2)
    
    src_pts_h = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    projected_pts_h = np.dot(H, src_pts_h.T).T
    projected_pts = projected_pts_h[:, :2] / projected_pts_h[:, 2, np.newaxis]
    
    residuals = np.linalg.norm(projected_pts - dst_pts, axis=1)

    return np.mean(residuals)

def explain(device, dataset, annot_0, annot_1, model, layer_keys, k_lines=10,k_colors=10, only_evaluate=False, return_img=False, match_img_save_path=None, color_map_save_path=None):
    img_0, name_0, _ = dataset.get_image_transformed(annot_0)
    img_1, name_1, _ = dataset.get_image_transformed(annot_1)

    img_0 = img_0.unsqueeze(0).to(device)
    img_1 = img_1.unsqueeze(0).to(device)

    H_lightglue = get_lightglue_homography(device, img_0, img_1)

    # TODO - change to get untransformed version straight from dataset
    
    # get intermediate feature maps + embeddings
    feature_maps_0, emb_0 = get_intermediate_feature_maps_and_embedding(img_0, model, layer_keys)
    feature_maps_1, emb_1 = get_intermediate_feature_maps_and_embedding(img_1, model, layer_keys)
            

    # backpropagate cosine similarity back to the intermediate layers
    emb_0.retain_grad()
    emb_1.retain_grad()
    
    cosine_sim = F.cosine_similarity(emb_0, emb_1, dim=1)
    cosine_sim.backward()

    intermediate_relevances_0 = get_intermediate_relevances(img_0, emb_0.grad, model, layer_keys)
    intermediate_relevances_1 = get_intermediate_relevances(img_1, emb_1.grad, model, layer_keys)

    results = {'annots': (str(annot_0), str(annot_1)),
               'cosine_sim': cosine_sim.item(),
               'is_match': 1 if name_0 == name_1 else 0,
               'scores': {}}

    for layer_key in layer_keys:
        feature_map_0 = feature_maps_0[layer_key]
        feature_map_1 = feature_maps_1[layer_key]
        intermediate_relevance_0 = intermediate_relevances_0[layer_key]
        intermediate_relevance_1 = intermediate_relevances_1[layer_key]
        
        # get a set of feature matches
        matches = get_feature_matches(feature_map_0, feature_map_1, img_0, img_1, match_img_save_path)

        matched_relevance = 0
    
        # go through matches and record each match's calculated relevance
        for match in matches:
            i0, j0 = match['coord0']
            i1, j1 = match['coord1']
            matched_relevance += intermediate_relevance_0[j0][i0]
            matched_relevance += intermediate_relevance_1[j1][i1]
            match['relevance'] = intermediate_relevance_0[j0][i0] * intermediate_relevance_1[j1][i1]

        matches.sort(key = lambda x: -x['relevance'])

        # score based on proportion of relevance matched
        total_relevance = intermediate_relevance_0.sum() + intermediate_relevance_1.sum()
        match_relevance_score = matched_relevance / total_relevance
        proportion_matched = len(matches) / (feature_map_0.shape[-1] * feature_map_1.shape[-2])

        # scores based on how well points follow a homography
        # TODO - clean this up
        all_pts_0 = np.array([match['keypoint0'].pt for match in matches])
        all_pts_1 = np.array([match['keypoint1'].pt for match in matches])

        top_count = int(feature_map_0.shape[-1] * feature_map_1.shape[-2] * 0.05)
        top_pts_0 = np.array([match['keypoint0'].pt for match in matches[:top_count]])
        top_pts_1 = np.array([match['keypoint1'].pt for match in matches[:top_count]])

        try:
            H_all_deep_features, _ = cv2.findHomography(all_pts_0, all_pts_1, method=0)
            residual_mean_deep_features_all = calculate_residuals(H_all_deep_features, all_pts_0, all_pts_1)
        except:
            residual_mean_deep_features_all = None

        try:
            H_top_deep_features, _ = cv2.findHomography(top_pts_0, top_pts_1, method=0)
            residual_mean_deep_features_top = calculate_residuals(H_top_deep_features, top_pts_0, top_pts_1)
        except:
            residual_mean_deep_features_top = None

        if H_lightglue is not None:
            residual_mean_lightglue_all = calculate_residuals(H_lightglue, all_pts_0, all_pts_1)
            residual_mean_lightglue_top = calculate_residuals(H_lightglue, top_pts_0, top_pts_1)

        else:
            residual_mean_lightglue_all = None
            residual_mean_lightglue_top = None

        result_dict = {'proportion_relevance_matched': match_relevance_score,
                        'proportion_matched': proportion_matched,
                        'residual_mean_lightglue_all': residual_mean_lightglue_all,
                        'residual_mean_lightglue_top': residual_mean_lightglue_top,
                        'residual_mean_deep_features_all': residual_mean_deep_features_all,
                        'residual_mean_deep_features_top': residual_mean_deep_features_top}
        result_dict = {k: float(v) if v else None for k, v in result_dict.items()}

        results['scores'][layer_key] = result_dict
        
        if not only_evaluate:
            # for each selected feature match, backpropagate to the original image
            pixel_relevances_0 = []
            pixel_relevances_1 = []
            for match in matches[:k_colors]:
                pixel_relevances_0.append(get_pixel_relevance(device, img_0, match['coord0'], model, layer_key))
                pixel_relevances_1.append(get_pixel_relevance(device, img_1, match['coord1'], model, layer_key))

            if color_map_save_path:
                np.save(f'{color_map_save_path}_1.npy', np.array(pixel_relevances_0, dtype=object), allow_pickle=True)
                np.save(f'{color_map_save_path}_2.npy', np.array(pixel_relevances_1, dtype=object), allow_pickle=True)
                

            # build visualized outputs
            output_img = draw_matches_and_color_maps(img_0, img_1, matches[:k_lines],
                                        intermediate_relevance_0, intermediate_relevance_1,
                                        pixel_relevances_0, pixel_relevances_1)

            if not return_img:
                print(f'Layer: {layer_key}')
            
                plt.figure(figsize=(50,50))
                plt.imshow(output_img)
                plt.axis('off')
                plt.show()

    if return_img:
        assert not only_evaluate, "can't return image if set to only evaluate"
        assert len(layer_keys) == 1, "can't return an output image with >1 layer key provided"
        return results, output_img

    return results

    
