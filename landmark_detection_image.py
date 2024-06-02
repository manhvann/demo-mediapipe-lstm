import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from make_landmarks import draw_landmarks_on_image

import cv2

# STEP 2: Create a PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("D:\\demo-mediapipe\\image\\yoga.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# Convert the image from RGB to BGR
annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# Display the image in a window
cv2.imshow("Annotated Image", annotated_image_bgr)

# Wait for the user to press 'q' to close the window
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
