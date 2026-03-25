'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    # <<<<< Implementation begin >>>>>

    # Extracting and preparing images for Kornia

    # Get images from dict
    img_list = list(imgs.values())
    img1 = img_list[0].float() / 255.0
    img2 = img_list[1].float() / 255.0
    
    # Get og dimensions 
    _, h, w = img1.shape

    # Adding batch size dimension
    img1_b = img1.unsqueeze(0)
    img2_b = img2.unsqueeze(0) 
    # Convert to grayscale
    img1_gr = K.color.rgb_to_grayscale(img1_b)
    img2_gr = K.color.rgb_to_grayscale(img2_b)

    # Extracting SIFT features
    sift = K.feature.SIFTFeature(num_features=2000)
    frames1, scores1, feats1 = sift(img1_gr)
    frames2, scores2, feats2 = sift(img2_gr)
    # Matching 
    dists, idxs = K.feature.match_snn(feats1[0], feats2[0], th=0.8)
    pts1 = K.feature.get_laf_center(frames1)[:, idxs[:, 0], :] # (1, N, 2)
    pts2 = K.feature.get_laf_center(frames2)[:, idxs[:, 1], :] # (1, N, 2)

    # Finding homography
    ransac = K.geometry.ransac.RANSAC('homography', inl_th=2.0, max_iter=2000)
    H, mask = ransac(pts2[0], pts1[0])

    # Calculating canvas size
    corners1 = torch.tensor([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=torch.float32).to(H.device)
    corners2 = torch.tensor([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=torch.float32).to(H.device)
    warped_corners2 = K.geometry.transform_points(H.unsqueeze(0), corners2)
    # Calculating total bounding box
    all_corners = torch.cat([corners1, warped_corners2], dim=1)
    min_xy = all_corners.min(dim=1)[0]
    max_xy = all_corners.max(dim=1)[0]
    # Calculating output dimensions
    out_w = int(max_xy[0, 0] - min_xy[0, 0])
    out_h = int(max_xy[0, 1] - min_xy[0, 1])

    # Shifting so min_xy at (0, 0)
    translation = torch.eye(3).to(H.device)
    translation[0, 2] = -min_xy[0, 0]
    translation[1, 2] = -min_xy[0, 1]

    # Final transormation for both img
    H1 = translation.unsqueeze(0)
    H2 = (translation @ H).unsqueeze(0)

    # Warp images onto canvas
    canvas_size = (out_h, out_w)
    img1_warped = K.geometry.transform.warp_perspective(img1_b, H1, canvas_size)
    img2_warped = K.geometry.transform.warp_perspective(img2_b, H2, canvas_size)

    # Blending to remove objects 
    mask1 = (img1_warped.sum(dim=1, keepdim=True) > 0).float()
    mask2 = (img2_warped.sum(dim=1, keepdim=True) > 0).float()

    
    stitched = (img1_warped + img2_warped) / (mask1 + mask2 + 1e-6)
    img = (stitched.squeeze(0) * 255.0).byte()
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    # <<<<< Implementation begin >>>>>

    # Setting up 
    fnames = sorted(list(imgs.keys()))
    n = len(fnames)
    img_list = [imgs[f].float() / 255.0 for f in fnames]
    device = img_list[0].device

    sift = K.feature.SIFTFeature(num_features=2000)
    all_frames, all_feats = [], []

    # Extract features for every image in the set
    for img in img_list:
        img_b = img.unsqueeze(0)
        img_gr = K.color.rgb_to_grayscale(img_b) 
        f, s, d = sift(img_gr)
        all_frames.append(f)
        all_feats.append(d)

    # Comparing pairs to find overlap
    overlap_mtx = torch.eye(n).to(device)
    H_rel = {}
    ransac = K.geometry.ransac.RANSAC('homography', inl_th=2.0, max_iter=2000)

    for i in range(n):
        for j in range(i+1, n):
            dists, idxs = K.feature.match_snn(all_feats[i][0], all_feats[j][0], th=0.8)

            if len(idxs) > 30:
                pts_i = K.feature.get_laf_center(all_frames[i])[:, idxs[:,0], :]
                pts_j = K.feature.get_laf_center(all_frames[j])[:, idxs[:,1], :]

                try:
                    H_ji, mask = ransac(pts_j[0], pts_i[0])

                    # confirm if enough consistent points found
                    if mask.sum() > 20: 
                        overlap_mtx[i, j] = 1
                        overlap_mtx[j, i] = 1
                        H_rel[(j, i)] = H_ji
                        H_rel[(i, j)] = torch.inverse(H_ji)
                except: 
                    continue

    # Globally aligning (BFS)
    H_global = {0: torch.eye(3).to(device)}

    queue = [0]
    visited = {0}

    while queue:
        curr = queue.pop(0)
        for neighbor in range(n):
            if overlap_mtx[curr, neighbor] == 1 and neighbor not in visited:
                H_global[neighbor] = H_global[curr] @ H_rel[(neighbor, curr)]

                visited.add(neighbor)
                queue.append(neighbor) 

    # Canvas calc
    all_corners_warped = []
    for i in H_global.keys():
        _, h, w = img_list[i].shape
        corners = torch.tensor([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=torch.float32).to(device)
        warped = K.geometry.transform_points(H_global[i].unsqueeze(0), corners)
        all_corners_warped.append(warped)

    all_corners_warped = torch.cat(all_corners_warped, dim=1)
    min_xy = all_corners_warped.min(dim=1)[0]
    max_xy = all_corners_warped.max(dim=1)[0]
    out_w = int(torch.ceil(max_xy[0, 0] - min_xy[0, 0]).item())
    out_h = int(torch.ceil(max_xy[0, 1] - min_xy[0, 1]).item())

    # Shifting  to stay in positive quadrant
    translation = torch.eye(3).to(device)
    translation[0, 2] = -min_xy[0, 0]
    translation[1, 2] = -min_xy[0, 1]

    # Final warping and stitching 
    canvas_size = (out_h, out_w)
    final_img = torch.zeros((1, 3, out_h, out_w)).to(device)
    final_mask = torch.zeros((1, 1, out_h, out_w)).to(device)

    for i in H_global.keys():
        H_final = (translation @ H_global[i]).unsqueeze(0)
        warped = K.geometry.transform.warp_perspective(img_list[i].unsqueeze(0), H_final, canvas_size)
        mask = (warped.sum(dim=1, keepdim=True) > 0).float()

        final_img += warped
        final_mask += mask

    stitched = final_img / (final_mask + 1e-6)
    panorama_img = (stitched.squeeze(0) * 255.0).byte()
    
    img = panorama_img
    overlap = overlap_mtx.to(torch.int)

    return img, overlap
