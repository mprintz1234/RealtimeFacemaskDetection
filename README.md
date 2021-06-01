# Realtime Facemask Detection
This application is a realtime facemask & facemask type detector. It utilizes two CNN models to predict (1) whether someone is wearing a mask and (2)
what type of mask they are wearing. Then, opencv is used to feed realtime video feed for prediction. 

## 1. How to run

#### 1.1 Downloads
Prepare the environment using conda 
```
source create -n facemask_detect python=3.6
source activate facemask_detect
pip install -r requirements.txt
```

#### 1.2 Start Program
Write into your terminal:
```
python main_video.py
```
The video feed should then appear on your desktop. To quit out of the video feed, press the "q" button on your keyboard

## 2. Model & Application Description
#### 2.1 Model Training

##### facemask detection
The facemask detection model was trained on the [Kaggle Face Mask Detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection). The model was built using PyTorch, and the training code can be found in the `/Model_1_Training.ipynb` jupyter notebook. 
The classification targets of this model are:
1. Wearing mask
2. Not wearing mask
3. Wearing a mask improperly

##### facemask type detection
The facemask type detection model was trained on a subset of the [Flickr-Faces-HQ Dataset](https://github.com/NVlabs/ffhq-dataset). Then, an application called [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace) was used to superimpose different types of masks onto the faces. The training code for this model can be found at `/Model_2_Training.ipynb`.
The classification targets of this model are:
1. Surgical mask
2. N95 mask
3. Cloth mask

#### 2.2 Live detection application
The `/main_video.py` handles the live video feed and any annotations. 
The `/utils.py` file contains the logic for preprocessing images and generating model prediction. In this file, we additionally used OpenCV's Caffe model to detect and extract faces for input into our two facemask models.