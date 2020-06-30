# Project: Deploy a People Counter at the Edge :guardsman:


<img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/simpleicons.svg" /> <ins> **GOAL:** </ins></br> Investigate different pre-trained models for person detection, and detect the number of people in the frame, and the time spent there.</br></br>

For this project, you’ll first find a useful person detection model and convert it to an Intermediate Representation for use with the Model Optimizer. Utilizing the Inference Engine, you'll use the model to perform inference on an input video, and extract useful data concerning the count of people in frame and how long they stay in frame. You'll send this information over MQTT, as well as sending the output frame, in order to view it from a separate UI server over a network.

In this project, you will utilize the Intel® Distribution of the OpenVINO™ Toolkit to build a People Counter app, including performing inference on an input video, extracting and analyzing the output data, then sending that data to a server. The model will be deployed on the edge, such that only data on 1) the number of people in the frame, 2) time those people spent in frame, and 3) the total number of people counted are sent to a MQTT server; inference will be done on the local machine.

You will also create a write-up comparing the performance of the model before and after use of the OpenVINO™ Toolkit, as well as examine potential use cases for your deployed people counter app.

In completing the project, you’ll get hands-on experience optimizing models for inference with the OpenVINO™ Toolkit, as well as building skills handling and extracting useful information from the deployed models.
