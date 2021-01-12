function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

diffValues=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
%diffValues=[0.1, 0.3, 1]';

error=[];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
m=10000000000;
ci=0;
si=0;

for i=(1:length(diffValues))
  for j=(1:length(diffValues))
    model= svmTrain(X, y, diffValues(i), @(x1, x2) gaussianKernel(x1, x2, diffValues(j)));
    pred=svmPredict(model,Xval);
    value=mean(double(pred ~= yval));
    error=[error; value];
%    fprintf("error\t\tC\t\tsigma\n");
%   fprintf("%d\t%d\t%d\n",value,diffValues(i),diffValues(j));
    if value<m
      m=value;
      ci=i;
      si=j;
    endif
    
  endfor
endfor


%fprintf("error\t\tC\t\tsigma\n");
%co=1;
%for i=(1:length(diffValues))
%  for j=(1:length(diffValues))
%    fprintf("%d\t%d\t%d\n",error(co),diffValues(i),diffValues(j));
%    co=co+1;
%   endfor
%endfor


C=diffValues(ci);
sigma=diffValues(si);




% =========================================================================

end
