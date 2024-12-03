import cv2 as cv
import numpy as np
import os

def show_image(img_bgr):
    cv.imshow('img', img_bgr)
    cv.waitKey(0)

def retrieve_images(path='data/'):
    files = os.listdir(path)
    imgs = []
    for f in files:
        if f[-3:] == 'jpg':
            img = cv.imread(os.path.join(path, f))
            imgs.append(img)
            print(img.shape)
    return imgs

def create_mask(img, thresh=10):
    b, g, r = cv.split(img)

    # Compute absolute differences between the channels
    diff_rg = cv.absdiff(r, g)
    diff_rb = cv.absdiff(r, b)
    diff_gb = cv.absdiff(g, b)

    # Create a mask where all differences are below the threshold
    mask = (diff_rg < thresh) & (diff_rb < thresh) & (diff_gb < thresh)

    # Convert mask to binary (0 or 255)
    mask = mask.astype(np.uint8) * 255

    return mask

def find_keypoints(img1, img2):
    mask1 = create_mask(img1)
    mask2 = create_mask(img2)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    cv.imwrite('data/processed/1gray.jpg', gray1)
    cv.imwrite('data/processed/2gray.jpg', gray2)

    sift = cv.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    # img = cv.drawKeypoints(gray1, keypoints1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Use a brute-force matcher to find matches between descriptors
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    print('done2')

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('data/processed/sift_matches.jpg', img3)


if __name__ == '__main__':
    imgs = retrieve_images()
    img1 = imgs[0]
    # mask = create_mask(img1)
    # img1_masked = cv.bitwise_and(img1, img1, mask=mask)
    # cv.imwrite('data/img1_masked.jpg', img1_masked)
    img2 = imgs[1]
    find_keypoints(img1, img2)
