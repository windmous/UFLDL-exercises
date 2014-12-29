function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

M = size(data, 2);

z2 = bsxfun(@plus, W1 * data, b1);
a2 = sigmoid(z2);

z3 = bsxfun(@plus, W2 * a2, b2);
a3 = z3;

diffSquare = (a3 - data) .^ 2;
squareCost = (1 ./ M) * sum(0.5 .* sum(diffSquare, 1));
weightDecay = 0.5 * (sum(sum(W1 .^2)) + sum(sum(W2 .^2)));

avgRho = 1 ./ M * sum(a2, 2);
KLDivergence = sum(sparsityParam * log(sparsityParam ./ avgRho) + (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - avgRho)));
cost = squareCost + lambda * weightDecay + beta * KLDivergence;

%% ---------- calculate gradient --------------------------------------
delta3 = -(data - a3);
delta2 = (W2' * delta3 + beta .* (-(sparsityParam ./ repmat(avgRho, 1, M)) + ...
        (1 - sparsityParam) ./ (1 - repmat(avgRho,1, M)))) .* a2 .* (1 - a2);
    
W2grad(:) = delta3 * a2' ./ M + lambda .* W2;
b2grad(:) = sum(delta3, 2) ./ M;
W1grad(:) = delta2 * data' ./ M + lambda .* W1;
b1grad(:) = sum(delta2, 2) ./ M;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end