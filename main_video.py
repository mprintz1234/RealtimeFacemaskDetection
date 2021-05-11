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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""def drawBox(img, bbox, out_labels, score):
    if (len(bbox) != 0):
        probs = softmax(score[0].detach().numpy()[0])
        label_dict = {0: ['No Mask', (0, 0, 255)],
                    1: ['Wearing Mask', (0, 255, 0)],
                    2: ['Wearing Mask Improperly', (0, 255, 255)]}

        for bound, label in zip(bbox, out_labels):
            label_str = (label_dict[int(label)])[0] + '. Score: ' + "{:.2f}%".format(probs[int(label)] * 100)
            y = bound[1] - 10 if bound[1] - 10 > 10 else bound[1] + 10
            cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]),
                        (label_dict[int(label)])[1], 2)
            cv2.putText(img, label_str, (bound[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (label_dict[int(label)])[1], 2)

    return img"""

"""def drawBox3(img, bbox, out_labels, out_labels2, score):
    if (len(bbox) != 0):
        probs = softmax(score[0].detach().numpy()[0])
        label_dict = {0: ['No Mask', (0, 0, 255)],
                    1: ['Wearing Mask', (0, 255, 0)],
                    2: ['Wearing Mask Improperly', (0, 255, 255)]}

        label_dict_type = {0: ['N95 Mask', (0, 0, 255)],
                    1: ['Surgical Mask', (0, 255, 0)],
                    2: ['KN95 Mask', (0, 255, 255)],
                    3: ['Gas Mask', (0, 255, 255)],
                    4: ['Cloth Mask', (0, 255, 255)]}

        for bound, label, label2 in zip(bbox, out_labels, out_labels2):
            if (label == '0'):
                label_str = (label_dict[int(label)])[0] + '. Score: ' + "{:.2f}%".format(probs[int(label)] * 100)
            else:
                label_str = (label_dict[int(label)])[0] + ": {}".format((label_dict_type[int(label2)])[0]) + '. Score: ' + "{:.2f}%".format(probs[int(label)] * 100)
            y = bound[1] - 10 if bound[1] - 10 > 10 else bound[1] + 10
            cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]),
                        (label_dict[int(label)])[1], 2)
            cv2.putText(img, label_str, (bound[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (label_dict[int(label)])[1], 2)
    return img

def facemaskDetectImage(img, pil_image):
    # Load face detection model and detect faces
    net = cv2.dnn.readNetFromCaffe('models/caffe/deploy.prototxt.txt',
                                   'models/caffe/res10_300x300_ssd_iter_140000.caffemodel')
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Load facemask detection model
    transform0 = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         ])
    device = torch.device('cpu')
    model = torch.load('models/model_1_RGBNet_and_transform_best.pt', map_location=device)

    # Output bbox and labels for all faces in video stream
    bbox = []
    out_labels = []
    score = []

    # Draw rectangle using caffe
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # predict mask or no mask
            # print(type(pil_image))
            pil_cropped = pil_image.crop((startX, startY, endX, endY))
            pil_cropped = pil_cropped.convert('RGB')
            # pil_cropped.show()
            pil_trans = transform0(pil_cropped)
            pil_trans = pil_trans.unsqueeze(0)
            outputs = model(pil_trans.to(device))
            preds = torch.argmax(outputs)
            label = str(preds.item())

            bbox.append([startX, startY, endX, endY])
            out_labels.append(label)
            score.append(outputs)

    return bbox, out_labels, score


def facemaskTypedetect(img, pil_image):
    net = cv2.dnn.readNetFromCaffe('models/caffe/deploy.prototxt.txt',
                                   'models/caffe/res10_300x300_ssd_iter_140000.caffemodel')
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Load model
    transform0 = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         ])

    device = torch.device('cpu')

    model = torch.load('models/model_2_pa2net_best.pt', map_location=device)

    # Output bbox and labels
    bbox = []
    out_labels = []
    score = []

    # Draw rectangle using caffe
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
  
            # predict mask or no mask
            #print(type(pil_image))
            pil_cropped = pil_image.crop((startX, startY, endX, endY))
            #pil_cropped = pil_cropped.convert('RGB')
            #pil_cropped.show()
            pil_trans = transform0(pil_cropped)
            pil_trans = pil_trans.unsqueeze(0)
            outputs = model(pil_trans.to(device))

            preds = torch.argmax(outputs)
            label = str(preds.item())

            bbox.append([startX, startY, endX, endY])
            out_labels.append(label)
            score.append(outputs)

    return bbox, out_labels, score   """


"""def loadImage(img_path):
    img = cv2.imread(img_path)
    pil_image = Image.open(img_path)
    return img, pil_image"""


"""
Draws the boundary boxes and model results over the input faces
"""
def drawBox(img, bbox, mask_results, type_results, models):
    if len(bbox) != 0:
        # mask_probs = softmax(mask_results[0].detach().numpy()[0])
        # type_probs = softmax(type_results[0].detach().numpy()[0])

        for bound, mask, _type in zip(bbox, mask_results, type_results):
            mask_score = mask[0]
            mask_label = mask[1]
            type_score = _type[0]
            type_label = _type[1]
            mask_probs = softmax(mask_score.detach().numpy()[0])

            if type_label == -1:
                label_str = (models.mask_labels[int(mask_label)])[0] + '. Score: ' + "{:.2f}%".format(
                    mask_probs[int(mask_label)] * 100)
            else:
                type_probs = softmax(type_score.detach().numpy()[0])
                label_str = (models.mask_labels[int(mask_label)])[0] + ": {}".format(
                    (models.type_labels[int(type_label)])[0]) + '. Score: ' + "{:.2f}%".format(
                    mask_probs[int(mask_label)] * 100)
            y = bound[1] - 10 if bound[1] - 10 > 10 else bound[1] + 10
            cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]),
                          (models.mask_labels[int(mask_label)])[1], 2)
            cv2.putText(img, label_str, (bound[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (models.mask_labels[int(mask_label)])[1], 2)
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

    # Draw rectangle using caffe
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
    device = torch.device('cpu')
    models = ModelUtils('models/caffe/deploy.prototxt.txt',
                        'models/caffe/res10_300x300_ssd_iter_140000.caffemodel',
                        'models/model_1_RGBNet_and_transform_best.pt',
                        'models/model_2_pa2net_best.pt',
                        transform_mask,
                        transform_type,
                        device)

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
        #cv2.imshow("edited_img", edited_img)
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
