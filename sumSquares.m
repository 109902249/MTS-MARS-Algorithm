function value=sumSquares(X)
%------------------------------------------------------------
% 'sumSquares' is the sum squares function
% for Nonlinear Optimization.
% https://www.sfu.ca/~ssurjano/sumsqu.html
%
% domain: -20<=x_i<=20 for i=1,...,d                  
%                  
% f_max = -1
% x_max = zeros(d,1)   
%-----------------------------------------------------------
D=size(X);
A=X.^2;
B=zeros(D(1),1);
for i=1:D(1)
    B(i)=i;
end
C=B*ones(1,D(2)).*A;
value=sum(C)+1;
value=-value;
end