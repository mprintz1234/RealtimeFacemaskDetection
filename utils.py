import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

"""
Model Util Class
loading model, img preprocessing, and prediction
"""
class ModelUtils:
    # Initialize the models and necessary preprocessing
    def __init__(self, face_detect_pth, face_detect_weights, mask_detect_pth, type_detect_pth, m_transform, t_transform,
                 device):
        self.device = device

        # Load face detection model
        self.face_detect = cv2.dnn.readNetFromCaffe(face_detect_pth, face_detect_weights)

        # Load mask detection model
        self.mask_detect = torch.load(mask_detect_pth, map_location=device)
        self.mask_transform = m_transform
        self.mask_labels = {0: ['No Mask', (0, 0, 255)],
                            1: ['Wearing Mask', (0, 255, 0)],
                            2: ['Wearing Mask Improperly', (0, 255, 255)]}

        # Load type detection model
        self.type_detect = torch.load(type_detect_pth, map_location=device)
        self.type_transform = t_transform
        self.type_labels = {0: ['N95 Mask', (0, 0, 255)],
                            1: ['Surgical Mask', (0, 255, 0)],
                            2: ['KN95 Mask', (0, 255, 255)],
                            3: ['Gas Mask', (0, 255, 255)],
                            4: ['Cloth Mask', (0, 255, 255)]}

    # Get all faces boundary boxes from image
    def getFaces(self, img):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_detect.setInput(blob)
        detections = self.face_detect.forward()

        return detections, w, h

    # Predict if face is wearing mask
    def getMasksModel(self, pil_image, is_rgb=True):
        # Convert image to RGB and crop to 32x32
        if is_rgb:
            pil_image = pil_image.convert('RGB')
        pil_trans = self.mask_transform(pil_image)
        pil_trans = pil_trans.unsqueeze(0)

        # Predict on transformed image
        outputs = self.mask_detect(pil_trans.to(self.device))
        preds = torch.argmax(outputs)
        label = str(preds.item())

        return outputs, label

    # Predict the type of mask the face is wearing
    def getTypeModel(self, pil_image, is_rgb=True):
        # Convert image to RGB and crop to 32x32
        if is_rgb:
            pil_image = pil_image.convert('RGB')
        pil_trans = self.type_transform(pil_image)
        pil_trans = pil_trans.unsqueeze(0)

        # Predict on transformed image
        outputs = self.type_detect(pil_trans.to(self.device))
        preds = torch.argmax(outputs)
        label = str(preds.item())

        return outputs, label


"""
Compute softmax values for each sets of scores in x.
"""
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


"""
Baseline model adapted from COMP4211 Spring 2021 PA2
"""
class PA2Net(nn.Module):
    def __init__(self, first_in_channel=1):
        super(PA2Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=first_in_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(512)
        self.pool2 = nn.AvgPool2d(16, 16, padding=0)

        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        self.fully_connected = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        # Convolution layers for the images
        x = F.relu(self.norm1(self.conv1(x)))  # 32, 32, 32
        x = F.relu(self.norm2(self.conv2(x)))  # 32, 32, 32
        x = self.pool1(x)  # 32, 16, 16
        x = F.relu(self.norm3(self.conv3(x)))  # 64, 16, 16
        x = F.relu(self.norm4(self.conv4(x)))  # 128, 16, 16
        x = F.relu(self.norm5(self.conv5(x)))  # 256, 16, 16
        x = F.relu(self.norm6(self.conv6(x)))  # 512, 16, 16
        x = self.pool2(x)  # 512, 1, 1
        x = x.view(-1, 512)

        # Full connected layers and ouptut
        x = self.fully_connected(x)
        return x
