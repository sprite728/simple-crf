% State feature
% input: yi, x, i, o
% yi: label at i
% x: input sequence
% i: input position
% o: off set
%
% output:
% state feature (k*1)
%
% Chang Gong
% 2014-11-29

function f = fStateFeatureTest(yi, x, i, o)
k=2;
f = zeros(k,1);

% feature 1
x1 = x(o+i)>0.5;
f(1) = (x1 == yi);
% feature 2
x3 = mean(x(o+i-1:o+i+1))>0.5;
f(2) = (x3 == yi);
% feature 3
x5 = mean(x(o+i-2:o+i+2))>0.5;
f(3) = (x5 == yi);

end