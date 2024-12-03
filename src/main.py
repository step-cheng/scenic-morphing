import cv2 as cv
import numpy as np
import os
import math
from keypoints import *

def show_image(img_bgr):
    cv.imshow('img', img_bgr)
    cv.waitKey(0)

def retrieve_images(path='data/'):
    files = os.listdir(path)
    imgs = []
    for f in files:
        if f[-3:] == 'jpg':
            img = cv.imread(os.path.join(path, f))
            img = cv.resize(img, (720,960))
            imgs.append(img)
    return imgs

def get_points(imgs):
    kps = []
    for img in imgs:
        points = select_points_from_image(img)
        kps.append(points)

    return kps

def get_homography(ref, next):
    A = []
    b = []
    for (x,y), (x_, y_) in zip(ref, next):
        A.append([x,y,1,0,0,0,-x*x_, -y*x_])
        A.append([0,0,0,x,y,1,-x*y_, -y*y_])
        b.append(x_ - x)
        b.append(y_ - y)

    A = np.array(A)
    b = np.array(b)
    h = np.linalg.lstsq(A,b,rcond=None)[0]
    H = np.reshape(np.append(h,0),(3,3)) + np.eye(3)
    # ref = np.array(ref)
    # next = np.array(next)
    # print(ref, next)
    # H, _ = cv.findHomography(ref, next)
    return H

def blend_image_pair(img1,img2,num_frames=30):
    # return 10 images
    frames = []
    for t in np.linspace(0,1,num_frames):
        f = cv.addWeighted(img1, 1 - t, img2, t, 0)
        # f = np.clip(f,0,255).astype(np.uint8)
        frames.append(f)
    return frames

def warp_inpaint(img, ref, H):
    height,width = img.shape[:2]
    img_w = cv.warpPerspective(img, H, (width,height))
    # mask = (img_w == 0).all(axis=2).astype(np.uint8)
    # img_w = cv.inpaint(img_w, mask, inpaintRadius=3)

    # mask = (img_w == 0).all(axis=2)
    # img_w[mask] = ref[mask]

    # img_w = cv.copyMakeBorder(img_w, 50, 50, 50, 50, cv.BORDER_REPLICATE)
    return img_w

def crop_all(imgs, crop=None):
    if not crop:
        x1,y1,x2,y2 = -math.inf, -math.inf, math.inf, math.inf
        for img in imgs:
            roi = cv.selectROI("select", img)
            x,y,w,h = map(int,roi)
            x1 = max(x1,x)
            y1 = max(y1,y)
            x2 = min(x2,x+w)
            y2 = min(y2,y+h)
        print(x1,y1,x2,y2)
    else:
        x1,y1,x2,y2 = crop
    imgs_crop = []
    for img in imgs:
        img_crop = img[y1:y2, x1:x2]
        imgs_crop.append(img_crop)
        show_image(img_crop)
    return imgs_crop


def run(kps=None, crop=None):
    imgs = retrieve_images()
    if not kps:
        kps = get_points(imgs)
    
    n = len(imgs)
    imgs_w = [imgs[0]]
    for i in range(1,n):
        H = get_homography(kps[i], kps[0])
        img_w = warp_inpaint(imgs[i], imgs[0], H)
        imgs_w.append(img_w)
    imgs_final = crop_all(imgs_w, crop)

    video_frames = []
    for i in range(1,n):
        video_frames.extend(blend_image_pair(imgs_final[i-1],imgs_final[i]))
    
    height,width = video_frames[0].shape[:2]
    video_path = 'scenic_morphing.mp4'
    out = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'mp4v'),10,(width,height))

    for f in video_frames:
        out.write(f)
    
    out.release()
    print("Done")



if __name__ == '__main__':
    kps = [[(100,979),(97,567),(395,242),(694, 565),(696, 977)],
           [(91, 964),(93, 549),(406, 230),(717, 554),(713, 966)],
           [(83, 968),(83, 579),(395, 276),(706, 586),(704, 973)],
           [(110, 978),(106, 580),(397, 276),(686, 587),(683, 974)],
           [(96, 986),(94, 597),(391, 289),(688, 599),(686, 990)],
           [(106, 972),(102, 587), (398, 286),(696, 586),(697, 973)],
           [(117, 978),(110, 596),(392, 296), (684, 593), (688, 975)],
           [(103, 978),(96, 589),(390, 284),(694, 588),(697, 978)]
           ]
    crop = (22,17,693,944)
    run(kps=kps, crop=crop)
    # imgs = retrieve_images()
    # # mask = create_mask(img1)
    # # img1_masked = cv.bitwise_and(img1, img1, mask=mask)
    # # cv.imwrite('data/img1_masked.jpg', img1_masked)
    # kps = get_points(imgs[:2])
    # homos = get_homography(imgs, kps)
