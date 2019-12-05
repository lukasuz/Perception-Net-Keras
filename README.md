# Perception Net Keras
This is an implementation of PerceptionNet for multimodal human activity recognition. The perception_net.py yields the original model with optional dilated convolutions. The perception_net_separate_channel.py yields a modified model with separate convolutions for each sensor channel before fusion in the third convolution. The prep_data.py is based on the UCI HAR [https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones] data set. The experiment.py file shows how the train function can be called.


### References
"PerceptionNet: A Deep Convolutional Neural Network for Late Sensor Fusion" by Kasnesis and Patrikakis [https://arxiv.org/abs/1811.00170]

