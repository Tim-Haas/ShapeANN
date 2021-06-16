# ShapeANN
ShapeANN is a deep-learning algorithm to approximate the shape of a three-dimensional ellipsoid, describing a bubble, by two perpendicular projections.
For this purpose, ellipse parameters and the rising path must first be extracted from the projections using computer vision. 
Then this feature input vector is processed by a neural network which estimates the ellipsoid axes A, B and C with an accuracy of 99.9 %. 

In this repository you will find five trained ANNs. These can either be used individually or for the ensemble method (reduces outliers, longer runtime).

For more detailed information check the corresponding paper: ShapeANN: 3D Bubble Shape Reconstruction by an Artificial Neural Networks' (not yet published)

If you need further help you can contact me (haas@iob.rwth-aachen.de)

<b>Example code single ANN </b>

    %load a network
    load('net_1400.mat');
    %assemble an input vector
    input=[b1,b2,a1,a2,phi1,phi2,α,β]; %please refer to the paper
    %predict the axes
    shape = predict(net,input);




<b>Example code ensemble method</b>

    %load the networks

    network(1)=load('net_1000.mat');
    network(2)=load('net_1100.mat');
    network(3)=load('net_1200.mat');
    network(4)=load('net_1300.mat');
    network(5)=load('net_1400.mat');
    
    %assemble an input vector
    
    input=[b1,b2,a1,a2,phi1,phi2,α,β]; %please refer to the paper

    %Predict the axes with all networks

    for i=1:5
    shape(:,:,i) = predict(network(i).net,input); 
    end

    %compute the mean prediction

    ensembleaxis=mean(shape,3);

Copyright (c) Institut für Industrieofenbau RWTH Aachen University - All Rights Reserved Unauthorized copying of this file, via any medium is strictly prohibited Proprietary and confidential Written by Tim Haas haas@iob.rwth-aachen.de, 2021

Requires MATLAB R2020b or later and the neural_network_toolbox and statistics_toolbox
