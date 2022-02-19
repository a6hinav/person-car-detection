# person-car-detection
This repository contains a trained model to detect car and person in an image.

For this use-case, I've proceeded with the transfer learning approach. A pre-trained model is used for training the data. This model is trained using the tensorflow 1.14 object detection api.

# Model Name
The pre-trained model used for training the dataset is FRCNN model trained on COCO Dataset.

# Links
FRCNN pre-trained model - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
Link to dataset - https://www.google.com/url?sa=D&q=https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz&ust=1645362840000000&usg=AOvVaw1tmChGtzJmuIkozIAe7fIA&hl=en&source=gmail

# FRCNN MODEL
FRCNN or Faster R-CNN is an object detection model that uses RPN for finding out all possible regions in an image where an object can be present.

![image](https://user-images.githubusercontent.com/100028415/154803140-695a08f5-c9d7-46f7-81b1-9cafc14fca22.png)

RPN gives feature maps of all the anchor boxes which belongs to the object class and passes it to the ROI pooling layer. This ROI pooling layer then flattens the feature maps and pass it to the regressor and classifier which are used for classifying the objects and drawing a bounding box on it.

# Approach
For training the model, I have first modified the dataset by converting the coco format annotations into pascal voc format annotations for all the images. Then generated the tfrecord files for training and testing from this data and started the training for around 18000 steps. 
After the training, I freezed the graph from the last checkpoint and performed the inferences on that file.

# Primary Analysis
The model is able to predict even person and cars that are partially present in the image. But due to this there is a possibility of having more false positives in the results.

# Inferences
![testcase (8)](https://user-images.githubusercontent.com/100028415/154804914-46b4a2b9-3739-480a-a82b-116638021971.jpg)
![testcase (24)](https://user-images.githubusercontent.com/100028415/154804936-5474338f-068d-4686-82ac-f83334d484a6.jpg)
![testcase (18)](https://user-images.githubusercontent.com/100028415/154805001-dfdc1904-3a7f-4688-a0ce-d9c6e5e9836a.jpg)
![testcase (5)](https://user-images.githubusercontent.com/100028415/154805028-860f0f77-b71e-403c-8149-f06330572312.jpg)
![testcase (17)](https://user-images.githubusercontent.com/100028415/154805263-0f108bbe-8faa-4db6-a332-b4b764308ae1.jpg)

# False Positives
![testcase (2)](https://user-images.githubusercontent.com/100028415/154805087-dd4bf8aa-9a92-49fd-bb4f-a0022e5a9d8a.jpg)
![testcase (31)](https://user-images.githubusercontent.com/100028415/154805209-5d772f95-b182-4e34-a243-562ec4c15924.jpg)

# Conclusion
The model is able to detect cars and persons in an image even if they are partial or blurred. 

# Recommendations
If we reduce some partial objects in the dataset and take only the images in which more than 70% of the object can be seen then the model will be able to detect objects more accuratley and the false positives can also be reduced.
