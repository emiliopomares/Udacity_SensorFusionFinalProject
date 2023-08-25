To run the code, it is necessary to install the dependencies in requirements.txt; also, the detections files should be downloaded from https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing and placed in the results folder. Also, the pretrained models (darknet and fpn-resnet) should be downloaded from https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing and https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing and placed under tools/objdet_models/darknet/pretrained and tools/objdet_models/fpn_resnet/pretrained respectively.

Depending on the OS, you could get this error when trying to run the code:

'''If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).'''
 
 
 Option 2. (executing the command 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python') should be the easiest fix.
