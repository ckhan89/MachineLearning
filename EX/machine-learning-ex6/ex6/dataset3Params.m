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
C_list = [0.8,0.9,1.0,1.1,1.2,1.3,1.4];
sigma_list = [0.8,0.9,1.0,1.1,1.2,1.3,1.4] / 10;
errorMin = 100;
x1 = [1,2,1];
x2 = [0,4,-1];
for i=1:length(C_list)
  for j =length(sigma_list)
    Cm = C_list(1,i);
    sigmam = sigma_list(1,j);
    model = svmTrain(X, y, Cm, @(x1, x2) gaussianKernel(x1, x2, sigmam));
    y_predict = svmPredict(model,Xval);
    error = (yval-y_predict)'*(yval-y_predict);
    if (error < errorMin)
      errorMin = error;
      C = Cm;
      sigma = sigmam;
    endif
    
  end
end


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







% =========================================================================

end
