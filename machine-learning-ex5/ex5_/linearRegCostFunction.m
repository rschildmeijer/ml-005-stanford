function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X * theta;     % calculate the hypothesis (1 * 2 matrix)

J = sum((1/(2*m)) * (predictions - y) .^2)

%regularize
n = size(theta, 1) % #features
reg = 0;
for j = 2:n
    reg = reg + theta(j) * theta(j);
end
J = J + reg * (lambda/(2*m));
    
%calculate the gradients
grad(1) = (1/m) * sum((predictions - y)' * X(:, 1));
for j = 2:n
    grad(j) = (1/m) * sum((predictions - y)' * X(:, j)) + (lambda/m) * theta(j); %might be X'
end








% =========================================================================

grad = grad(:);

end
