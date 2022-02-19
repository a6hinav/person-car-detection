import numpy as np
import tensorflow as tf
import cv2 as cv
import argparse
import os
import re


# Arguements for providing the image directory path of testing images and setting the threshold
parser = argparse.ArgumentParser()
parser.add_argument('imgpath', help='path of image directory to be scored', type=str)
parser.add_argument('--confidence', help='minimum confidence threshold', type=float, default=0.3)

# Reading label files containg the classes
inp_file_path = "label_map.pbtxt"
file = open(inp_file_path, 'r')
file_content = file.read()
file_content = file_content.replace('\n', ' ')
file_content = file_content.replace('\t', ' ')

list_content = re.split("item *{ *id *: *([0-9]*) * name *: *['\"]([a-z_A-Z_0-9]+)['\"] *} *", file_content)
# print(list_content)
for match_item in range(list_content.count('')):
    list_content.remove('')
result_dict = dict()

for iterator in range(0, len(list_content), 2):
    result_dict[list_content[iterator]] = list_content[iterator+1]
print(result_dict)

args = parser.parse_args()
default_confidence = float(args.confidence)
img_path = args.imgpath

result_path = os.path.join(img_path, 'result_c')
if not os.path.exists(result_path):
    os.mkdir(result_path)


list_files = os.listdir(img_path)

list_jpg = []
output_file_list = []


def object_detection(image, filename):  # function to do inference on the images
    # Restore session  
    # Read and pre-process an image.
    img = cv.imread(image)
    
    rows = img.shape[0]
    cols = img.shape[1]

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': img.reshape(1, img.shape[0], img.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    print('number of detections', num_detections)
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        
        if score > default_confidence and str(classId) in result_dict.keys():
            print('score', score)
            class_name = result_dict[str(classId)]
           
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            
            if class_name.split('_')[0] == "car":
                color = (0, 250, 0)
            elif class_name.split('_')[0] == "person":
                color = (0, 0, 250)
            else:
                color = (250, 0, 0)

            cv.putText(img, class_name, (int(x-3), int(y-3)), cv.FONT_HERSHEY_SIMPLEX, .9, color)
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), color, thickness=2)
            
    cv.namedWindow('output', cv.WINDOW_NORMAL)
    cv.imshow('output', img)
    cv.waitKey(2)
    cv.imwrite(os.path.join(result_path, filename+'.jpg'), img)


# Reading the freezed graph file
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Creating the tensorflow session
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for file in list_files:
        if file[-4:].lower() == '.jpg':
            file_path = os.path.join(img_path, file)
            object_detection(file_path, file[:-4])

print('Evaluation done.')
