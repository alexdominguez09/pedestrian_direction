# Pedestrian Movement Direction Recognition Using Convolutional Neural Networks
This software helps you to generate the dataset used in the published paper: "Pedestrian Movement Direction Recognition Using Convolutional Neural Networks"

<img src="https://github.com/alexdominguez09/pedestrian_direction/blob/master/dataset_diagram.png">

# Code
The software is written in C++ under code::blocks 13. The project file is provided but is not required. You can use your own IDE or c++ favourite compiler. This code assumes it has OpenCV 2.4 installed. This program extract and saves images from a given video. The extracted images are pedestrian moving to right, left or front. It is based in HOG to detect a pedestrian, and optical flow to detect the pedestrian is moving. It then saves all or some of the images in the process.

Once you have your own dataset created, you will need to train a network with it for 3 different classes: front, left and right. A small network can be trained for that. 

See the paper for full details.

# Cited in
A. Dominguez-Sanchez, M. Cazorla and S. Orts-Escolano, "Pedestrian Movement Direction Recognition Using Convolutional Neural Networks," in IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 12, pp. 3540-3548, Dec. 2017.

doi: 10.1109/TITS.2017.2726140
Retrieved from [https://arxiv.org/abs/1704.03952](https://arxiv.org/pdf/1704.03952.pdf)

# Paper published in

http://rua.ua.es/dspace/bitstream/10045/74053/5/2017_Dominguez-Sanchez_etal_IEEE-TITS_revised.pdf

http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8006277&isnumber=8169696
