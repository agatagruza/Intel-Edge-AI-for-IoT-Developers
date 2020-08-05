#!/usr/bin/env python3
# Work based on https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131 
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
#CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
#DEVICE = "CPU"
# check main.py for CPU_EXTENSION and DEVICE

def all_layers_supported(engine, net, console_output=False):
    # If at least one layer is not supported, funtion will return False, otherwise True
    supported_layers = engine.query_network(net, device_name='CPU')
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        return False
    return True

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.net = None
        self.in_blob = None
        self.out_blob = None
        self.exec_network = None
    
    def load_model(self, model, CPU_EXTENSION, DEVICE, console_output= False):
        ### TODO: Load the model ###
        
        # Load the Inference Engine API
        self.plugin = IECore()
        
        #Load IR files into their related class
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        ## TODO: Check for supported layers ###     
        ### TODO: Add any necessary extensions ###
        if not all_layers_supported(self.plugin, self.net, console_output=console_output):
            self.plugin.add_extension(CPU_EXTENSION, DEVICE)
            
        self.exec_network = self.plugin.load_network(self.net, DEVICE)
        print("IR successfully loaded into Inference Engine.")
      
        # Get the input layer
        self.in_blob = next(iter(self.net.inputs)) # Iterate throught input
        self.out_blob = next(iter(self.net.outputs)) # Iterate through output
       
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        layer_shapes_disc = dict()
        for items in self.net.inputs:
            layer_shapes_disc[items] = (self.net.inputs[items].shape)
        return layer_shapes_disc
    

    def exec_net(self, net_input, request_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.request_handle = self.exec_network.start_async(
                request_id, 
                inputs=net_input)
        #self.exec_network.start_async(request_id=0,inputs={self.input_blob: image})
        return 


    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.request_handle.wait()
    

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.request_handle.outputs[self.out_blob]
