import sys
import cv2
import numpy as np

if (len(sys.argv) != 4) :
	sys.exit("openCVCircles.py <image-in> <image-out> <descriptorfile>")

# 1. Read your image
img = cv2.imread( sys.argv[1] )

# 2. Load/Parse your SIFT file
# Assuming a file format where each line is: x y scale orientation
# For this example, we generate them, but you would load from file:
data = np.loadtxt( sys.argv[3] )
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(gray, None)

# If loading from a file, convert to keypoints like this:
keypoints = []
for row in data:
	kp = cv2.KeyPoint(x=row[0], y=row[1], size=row[2], angle=row[3])
	keypoints.append(kp)

# 3. Draw keypoints as circles
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS draws the circle and orientation
output_img = cv2.drawKeypoints(img, keypoints, None, (255,0,0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 4. Display or Save
cv2.imshow('SIFT Features', output_img)
cv2.imwrite( sys.argv[2], output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

