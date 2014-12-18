function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 1e-4;

for i = 1 : size(theta)
   fprintf('Begin checking %d-th parameter\n', i)
    
   rtheta = theta;
   rtheta(i) = rtheta(i) + EPSILON;
   
   ltheta = theta;
   ltheta(i) = ltheta(i) - EPSILON;
   
   [rcost, rgrad] = J(rtheta);
   [lcost, lgrad] = J(ltheta);
   
   numgrad(i) = (rcost - lcost) ./ (2 * EPSILON);
   
   fprintf('Finished checking %d-th parameter\n', i)
end


%% ---------------------------------------------------------------
end
