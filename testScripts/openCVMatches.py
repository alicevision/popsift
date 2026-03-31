import sys
import cv2
import numpy as np

if (len(sys.argv) != 5) :
	sys.exit("openCVMatches.py <image-in> <image-out> <popsiftDescriptorFile> <vlfeatDescriptorFile>")

# 1. Read your image
img = cv2.imread( sys.argv[1] )

# 2. Load/Parse your SIFT file
# Assuming a file format where each line is: x y scale orientation
# For this example, we generate them, but you would load from file:
data1 = np.loadtxt( sys.argv[3] )
data2 = np.loadtxt( sys.argv[4] )
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(gray, None)

kp1 = [cv2.KeyPoint(x=r[0], y=r[1], size=r[2], angle=r[3]) for r in data1]
des1 = np.ascontiguousarray(data1[:, 4:], dtype=np.float32)

kp2 = [cv2.KeyPoint(x=r[0], y=r[1], size=r[2], angle=r[3]) for r in data2]
des2 = np.ascontiguousarray(data2[:, 4:], dtype=np.float32)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match( des1, des2 );

print('Number of matches is {:d}'.format(len(matches)))

matches = sorted( matches, key = lambda x:x.distance )

# output_img = cv2.drawMatches( img, kp1, img, kp2, matches[:1000], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
output_img = cv2.drawMatches( img, kp1, img, kp2, matches[:1], None, matchColor=(0,0,255,127), singlePointColor=(255,0,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# 4. Display or Save
# cv2.imshow('SIFT Matches', output_img)
cv2.imwrite( sys.argv[2], output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

