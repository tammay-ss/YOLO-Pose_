This repo tries to use YOLO-Pose Estimation of Ultralytics to predict Carina points from X-rays and angiography images.
dataset_3 folder is the final dataset on which the YOLOv26s and YOLOv26m were trained seperately, the results and the jupyter notebook of training loop is present in folders: yolo-26m_metrics and yolo-26s_metrics.
If these results are analysed, YOLOv26s, right now, does a better job. 
The training data(dataset_3) comprises of 313 labelled images, which is small for Pose Estimation but good enough for medical images.
The predictions made on unseen 200 images, after the training, are stored in predicted_images_s and predicted_images_m respectively for both the models, 115 images were labelled by YOLOv26s, while 75 were labelled by YOLOv26m out of 200, this difference is because of the confidence the model possess.

conversion.py - script to convert JSON labels from label-studio to yolo format for pose estimation
conversion_2json.py -  script to convert the predicted labels (YOLO format) to JSON 
convert2p.py - script to conver the predicted and converted JSON labels to a predictions format accepted by label-studio 
