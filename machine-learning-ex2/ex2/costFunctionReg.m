function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
grad = zeros(size(theta));
h = sigmoid(X * theta);
theta(1) = 0;
J = -1 * ((y' * log(h)) + ((1 .- y)' * log(1 - h))) / m;

J = J + (lambda * sum(theta .^ 2) / (2 * m));

grad = (X' * (h - y)) ./ m;
theta = theta .* (lambda/m);
grad = grad + theta;

end
