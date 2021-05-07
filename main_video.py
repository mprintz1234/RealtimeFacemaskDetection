import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from utils import PA2Net
from PIL import Image
import os
from imutils.video import VideoStream
import imutils
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def loadImage(img_path):
    img = cv2.imread(img_path)
    pil_image = Image.open(img_path)
    return img, pil_image


def drawBox(img, bbox, out_labels, score):
    label_dict = {0: ['No Mask', (0, 0, 255)],
                  1: ['Wearing Mask', (0, 255, 0)],
                  2: ['Wearing Mask Improperly', (255, 0, 0)]}

    for bound, label in zip(bbox, out_labels):
        label_str = (label_dict[int(label)])[0]
        y = bound[1] - 10 if bound[1] - 10 > 10 else bound[1] + 10
        cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]),
                      (label_dict[int(label)])[1], 2)
        cv2.putText(img, label_str, (bound[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (label_dict[int(label)])[1], 2)

    return img

def facemaskDetectImage(img, pil_image):
    # Load image and run through face detection model
    # img = cv2.imread(img_path)
    # pil_image = Image.open(img_path)
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
         # transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         ])
    device = torch.device('cpu')
    """model = torch.load(
        '/Users/maximilian/Desktop/SideProjects/realtime_facemask_project/models/model_1_RGBNet_and_transform_best.pt',
        map_location=device)"""
    model = torch.load('models/model_1_RGBNet_best.pt', map_location=device)

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
            pil_cropped = pil_cropped.convert('RGB')
            #pil_cropped.show()
            pil_trans = transform0(pil_cropped)
            pil_trans = pil_trans.unsqueeze(0)
            outputs = model(pil_trans.to(device))

            """f = img[startY:endY, startX:endX]
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (32, 32))
            #cv2.imshow('f', f)
            #cv2.waitKey()
            # print(f.shape)
            f = np.moveaxis(f, -1, 0)
            f = torch.from_numpy(f)
            f = f.unsqueeze(0)
            print(f.shape)
            outputs = model(f.to(device).to(torch.float32))"""
            preds = torch.argmax(outputs)
            label = str(preds.item())

            bbox.append([startX, startY, endX, endY])
            out_labels.append(label)
            score.append(outputs)
            # label = str(preds) + str(outputs)

            # draw the bounding box of the face along with the associated
            # probability
            # text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(img, (startX, startY), (endX, endY),
            #              (0, 0, 255), 2)
            # cv2.putText(img, label, (startX, y),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Display the output
    # cv2.imshow('img', img)
    # cv2.waitKey()
    return bbox, out_labels, score


def videoRun():
    vs = VideoStream(src=0).start()
    time.sleep(2)

    while True:
        frame = vs.read()
        img = frame
        #img = cv2.imread(frame)
        rgb_arr = imutils.opencv2matplotlib(frame)
        pil_image = Image.fromarray(rgb_arr)
        bbox, out_labels, score = facemaskDetectImage(img, pil_image)
        edited_img = drawBox(img, bbox, out_labels, score)
        cv2.imshow("edited_img", edited_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    videoRun()

    """img_path = 'group_test.jpg'
    img, pil_image = loadImage(img_path)
    bbox, out_labels, score = facemaskDetectImage(img, pil_image)
    img = drawBox(img, bbox, out_labels, score)
    cv2.imshow('img', img)
    cv2.waitKey()"""
