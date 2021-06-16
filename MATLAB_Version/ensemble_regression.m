function outputs=ensemble_regression(network,inputs,method,varargin)
% ensemble_regression(network, inputs, method)
%
% ensemble_regression is a MATLAB function that employes different neural 
% networks (ANNs) to predict outputs based on the same input vector.
% The different outputs are combined by either computing its mean or mode.
%
%
% outputs: an array containing the output vectors 
%
% network: a struct that contains at least two pretrained regression ANNs 
%
% inputs:  an array containing the feature input vectors that are used to
%          make predictions
%
% method:  a string that defines whether the output should be the 'mean' or
%          'mode'
%
%
%
% Current Version:  01.06.2021
% Developped by:    Tim HAAS, IOB,RWTH Aachen University
%                   haas@iob.rwth-aachen.de
%


shape=zeros(size(inputs,1),size(network(1).net.Layers(end-1).Bias,1),size(network,2));
if nargin<3
        method='mean';
end
for i=1:size(network,2)
    shape(:,:,i) = predict(network(i).net,inputs);
end

if strcmp(method,'mode')
outputs=median(shape,3); 
else
outputs=mean(shape,3);
end

end
