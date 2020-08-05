# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## About statistics
Model has been run in Udacity environment. Model was running quite slow in the Udacity environment due to limited number of cores and memory 
(one core working at 2.3GHz and 4GB of memory in total)

## Explaining Custom Layers
One of the factors behind successful advancement in deep learning is an abundant variety and combination of layers that allow to create unique models. In OpenVino there is a list of predefined supported layers https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html   The process behind converting custom layers involves four steps:
1.	If a given layer doesn’t exist on the mentioned list, it is then recognized by a Model Optimizer as a custom layer. Before building the model’s internal representation Model Optimizer looks for every layer in known layers list. 
2.	Layers from the input model IR (Intermediate Representation) files is loaded by the Inference Engine into a specific device plugin. 
3.	Inference Engine will report an error if given layer is unsupported (not on a list of known layers for the device). Full list of the layers that are supported by each device plugin is available 
https://docs.openvinotoolkit.org/2019_R1.1/_docs_IE_DG_supported_plugins_Supported_Devices.html 
4.	Before using a custom layer, Model Extension generate templates for Model Optimizer. Created templates need to be updated by the user, compiled and then execute. 

More info https://docs.openvinotoolkit.org/latest/openvino_docs_HOWTO_Custom_Layers_Guide.html   

## Comparing Model Performance
I have searched Google for sample running the model I have used. I have found online documentation https://docs.openvinotoolkit.org/2019_R3.1/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html  that have a sample run script for YOLO model.  My steps to compare models before and after conversion to Intermediate Representations were as follow:

### Clone repo
git clone https://github.com/mystic123/tensorflow-yolo-v3.git 
cd tensorflow-yolo-v3 (has convert_weights.py and convert_weights_pb.py )

### Download COCO class names file: 
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

### Download model weights
wget https://pjreddie.com/media/files/yolov3.weights 

### Run a converter for YOLO-v3
python convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights 

### Convert YOLOv3 TensorFlow Model to the IR (from *.pb to *.bin and *.xml) by using model optimizer
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py  \
--input_model frozen_darknet_yolov3_model.pb \
--tensorflow_use_custom_operations_config \ /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json \
 --batch 1

### Running the video
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m tensorflow-yolo-v3/frozen_darknet_yolov3_model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm 

Note: Prior to making a write up, video was run using RCNN model, as in:
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Comparing Model Performance
The difference between model accuracy pre- and post-conversion to Intermediate Representations (IR) was considerable. 
-	~92% accuracy for YOLO-v3, detecting up to 5 people, low failing rate.
-	Much lower accuracy for YOLO-IR, detecting up to 7 people, much higher failing rate 

The size difference between the model pre- and post-conversion to Intermediate Representations (IR) was negligible. 

The inference time of the model pre- and post-conversion to Intermediate Representations (IR) was in favor of the OpenVino. OpenVino with IR finished 0.01 sec faster comparing to the original YOLO-v3.  

## Assess Model Use Cases
Some of the potential use cases of the people counter app are
-	Social distancing
-	Manufacturer 
-	Retail Stores
-	Supermarkets
-	Shopping Molls
-	Transportation
-	Surveillance
-	Museum and Galleries
-	Train Stations and Airports

Each of these use cases would be useful because it can
-	Measure store’s conversion ratio
-	Compare store performance across a geo located stores
-	Find unknown before patterns
-	Optimize any building layout and staffing levels
-	Improve customer service


## Assess Effects on End User Needs
Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows
-	Good lighting can increase a model performance and exposure details that were not visible before. Bad lighting can negatively impact the model due to the fact that details and colors of an image can get lost. 
-	Model accuracy can negatively impact a system performance by creating false positive and true negatives. 
-	Camera focal length/image size can impact a system performance and decrease model accuracy, depending on the camera angle and shadow.  
