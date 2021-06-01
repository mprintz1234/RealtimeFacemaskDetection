import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from utils import PA2Net, softmax, ModelUtils
from PIL import Image
import os
from imutils.video import VideoStream
import imutils
import time

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Draws the boundary boxes and write model results over the input faces
"""
def drawBox(img, bbox, mask_results, type_results, models):
    if len(bbox) != 0:
        # loop through all detected faces in the image
        for bound, mask, _type in zip(bbox, mask_results, type_results):
            mask_score, mask_label = mask
            type_score, type_label = _type

            # Calculate probability of mask detection
            mask_probs = softmax(mask_score.detach().numpy()[0])

            # If no mask is detected, only show no mask detected/mask worn incorrectly
            if type_label == -1:
                label_str = (models.mask_labels[int(mask_label)])[0] + '. Score: ' + "{:.2f}%".format(
                    mask_probs[int(mask_label)] * 100)

            # If mask detected, show both wearing mask and mask type
            else:
                type_probs = softmax(type_score.detach().numpy()[0])
                label_str = (models.mask_labels[int(mask_label)])[0] + ": {}".format(
                    (models.type_labels[int(type_label)])[0]) + '. Score: ' + "{:.2f}%".format(
                    mask_probs[int(mask_label)] * 100)

            # Put the boxes and text onto the cv2 image
            y = bound[1] - 10 if bound[1] - 10 > 10 else bound[1] + 10
            cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]),
                          (models.mask_labels[int(mask_label)])[1], 2)
            cv2.putText(img, label_str, (bound[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (models.mask_labels[int(mask_label)])[1], 2)

    # Return the edited image
    return img


"""
Detects the faces, masks, and mask type.
Return: face boundary boxes, mask detected, and mask type
"""
def maskDetection(img, pil_image, models):
    bbox = []
    mask_results = []
    type_results = []

    # Get face detections in img
    detections, w, h = models.getFaces(img)

    # Loop through each face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Predict facemask
            pil_cropped = pil_image.crop((startX, startY, endX, endY))
            mask_outputs, mask_labels = models.getMasksModel(pil_cropped, is_rgb=True)
            mask_results.append([mask_outputs, mask_labels])

            # Predict facemask type only if facemask detected
            if mask_labels == '1':
                type_outputs, type_labels = models.getTypeModel(pil_cropped, is_rgb=False)
            else:
                type_outputs = [0, 0, 0]
                type_labels = -1
            type_results.append([type_outputs, type_labels])

            bbox.append([startX, startY, endX, endY])

    return bbox, mask_results, type_results


"""
Starts the video stream and initializes all models
Begins infinite loop to loop through each frame of videostream to do prediction
"""
def videoRun():
    # Initialize video stream
    vs = VideoStream(src=0).start()

    # Initialize models
    transform_type = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         ])
    transform_mask = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         ])
    models = ModelUtils('models/caffe/deploy.prototxt.txt',
                        'models/caffe/res10_300x300_ssd_iter_140000.caffemodel',
                        'models/model_1_RGBNet_and_transform_best.pt',
                        'models/model_2_pa2net_best_may152021.pt',
                        transform_mask,
                        transform_type,
                        torch.device('cpu'))

    # Give video stream time to warm up
    time.sleep(2)

    # Enter infinite loop to get each frame of video stream
    while True:
        # Read in image from videostream
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        img = frame

        # Process images
        rgb_arr = imutils.opencv2matplotlib(frame)
        pil_image = Image.fromarray(rgb_arr)

        # Detect faces, mask, and mask type
        bbox, mask_results, type_results = maskDetection(img, pil_image, models)

        # Draw the results on the image
        edited_img = drawBox(img, bbox, mask_results, type_results, models)

        # Display the image
        cv2.imshow("edited_img", imutils.resize(edited_img, width=800))
        key = cv2.waitKey(1) & 0xFF

        # If q pressed, end video
        if key == ord("q"):
            break

    # clean up
    cv2.destroyAllWindows()
    vs.stop()


"""
Driver function of the streaming application
"""
if __name__ == '__main__':
    videoRun()
