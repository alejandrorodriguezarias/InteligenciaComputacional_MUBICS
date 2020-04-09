% henon map, E. Wan
function [x,y] = henon(NN)
a = 1.4;
b = 0.3;
x = zeros(NN+1,1);
y = zeros(NN+1,1);
for n = 1:NN,
x(n+1) = 1 - a*x(n)*x(n) + y(n);
y(n+1) =b*x(n);
end