import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

image = cv2.imread("ivan.jpg")
image_for_crop = cv2.imread("ivan.jpg")

result = detector.detect_faces(image)

bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0, 155, 255),
              2)

img_crop = image_for_crop[bounding_box[1]:bounding_box[1] + bounding_box[3],
           bounding_box[0]:bounding_box[0] + bounding_box[2]]

cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

cv2.imwrite("ivan_drawn.jpg", image)
cv2.imwrite("ivan_drawn_crop.jpg", img_crop)

print(result)
