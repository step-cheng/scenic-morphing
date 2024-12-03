import cv2 as cv
import numpy as np

def enlarge_keypoints(kp, scale=20):
    enlarged_keypoints = []
    for k in kp:
        print(k.size)
        # Create a new keypoint with the same properties but a larger size
        enlarged_kp = cv.KeyPoint(k.pt[0], k.pt[1], k.size * scale, k.angle,
                                   k.response, k.octave, k.class_id)
        enlarged_keypoints.append(enlarged_kp)
    return enlarged_keypoints

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

def sift_keypoints(img1, img2):
    mask1 = create_mask(img1)
    mask2 = create_mask(img2)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    cv.imwrite('data/processed/1gray.jpg', gray1)
    cv.imwrite('data/processed/2gray.jpg', gray2)

    sift = cv.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    # img = cv.drawKeypoints(gray1, keypoints1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)

    # Use a brute-force matcher to find matches between descriptors
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    print('done2')

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))
    kp1_large = enlarge_keypoints(kp1)
    kp2_large = enlarge_keypoints(kp2)
    img3 = cv.drawMatches(img1, kp1_large, img2, kp2_large, matches[:10], None, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv.imwrite('data/processed/sift_matches_mask.jpg', img3)


def select_points_from_image(image, num_points=5):
    """
    Allows the user to manually select a specified number of points on an image.

    Parameters:
        image_path (str): Path to the input image.
        num_points (int): Number of points to select.

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates of the selected points.
    """
    points = []  # List to store selected points

    def select_points(event, x, y, flags, param):
        """
        Mouse callback function to capture points on mouse click.
        """
        if event == cv.EVENT_LBUTTONDOWN:  # Left mouse button click
            points.append((x, y))  # Append the point
            print(f"Point selected: ({x}, {y})")
            # Draw the point on the image
            cv.circle(param, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            cv.imshow("Select Points", param)

    image_copy = image.copy()

    # Set up the OpenCV window and set the mouse callback function
    cv.imshow("Select Points", image_copy)
    cv.setMouseCallback("Select Points", select_points, param=image_copy)

    # Wait until the required number of points are selected
    print(f"Select {num_points} points by clicking on the image.")
    while len(points) < num_points:
        if cv.waitKey(1) & 0xFF == 27:  # Exit if 'Esc' is pressed
            print("Selection canceled.")
            points = []  # Clear the points
            break

    # Close the OpenCV window
    cv.destroyAllWindows()

    return points