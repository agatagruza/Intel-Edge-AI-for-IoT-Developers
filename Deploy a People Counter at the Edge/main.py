# Work based on https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60



def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser



def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    
    # Iniatilize variables for stats
    last_duration = 0
    total = 0
    period = 0
    
    current_count = 0
    last_count = 0
    time_length = 0 # Will calculate person/duration

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension, args.device)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.png') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    # Get and open video capture
    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    # out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    in_shape = net_input_shape['image_tensor']    
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()  # Read the next frame
        if not flag:
            break        
        key_pressed = cv2.waitKey(60) #'<'

        ### TODO: Pre-process the image as needed ###
        single_frame = cv2.resize(frame, (in_shape[3], in_shape[2])).transpose((2, 0, 1))
        single_frame = single_frame.reshape(1, *single_frame.shape)        

        ### TODO: Start asynchronous inference for specified request ###
        # Perform inference on the frame
        input_frame = {'image_tensor': single_frame,'image_info': single_frame.shape[1:len(single_frame.shape)]}
        infer_network.exec_net(input_frame, 0)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:  # Get the output of inference     
    
             # Write out the frame
             # input_frame.write(frame)

            ### TODO: Get the results of the inference request ###
            # result = infer_network.extract_output()
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
         
            refer = 0
            probs = result[0, 0, :, 2]
            for i, j in enumerate(probs): # loop over + automatic counter
                if j > args.prob_threshold:
                    refer += 1
                    bounding_box = result[0, 0, i, 3:]
                    point1 = (int(bounding_box[0] * width), int(bounding_box[1] * height))
                    point2 = (int(bounding_box[2] * width), int(bounding_box[3] * height))
                    frame = cv2.rectangle(frame, point1, point2, (0, 255, 0), 3)
                    
            if refer == current_count:
                period += 1
                if period >= 3: # For handling incorrect detections
                    if period == 3 and current_count < last_count:
                        time_length = int(time.time() - start_time) 
                    elif period == 3 and current_count > last_count: # New person walking in
                        total += current_count - last_count
                        start_time = time.time()  
            else:
                last_count = current_count
                current_count = refer
                if period > 2:
                    last_duration = period
                    period = 0 # reset "period"
                else:
                    period = last_duration + period
                    last_duration = 0 # reset "last_duration"        

                        
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person',payload=json.dumps({'count': current_count, 'total': total}))
            if time_length > 0:
                client.publish('person/duration',payload=json.dumps({'duration': time_length}))

        ### TODO: Send the frame to the FFMPEG server ###
        ### TODO: Write an output image if `single_image_mode` ###
        # Changing the size of the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        # Break if escape key pressed
        if key_pressed == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows() 
    
    #Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()