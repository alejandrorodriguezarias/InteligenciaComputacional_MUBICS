%Trasnform a series data into INPUTS and OUTPUTS considering a sliding
%window of size d
%inputs:
%   data: [N x 1 ] matrix with the data to be transformed
%   d: the size of the sliding window (positive integer)
%outputs:
%   INPUTS: [N-d x d] matrix containing the inputs 
%   OUTPUT: [N-d x 1] matrix containing the outputs

function [INPUTS, OUTPUTS]=slidingwindow(data,d);

%Number of samples
N=length(data);
OUTPUTS=data([d+1:N]); 
INPUTS=data([1:N-d]); %first column

%rest of columns
for i=1:d-1
    INPUTS=[INPUTS data([1+i:N-d+i])];
end
