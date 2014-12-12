% Transition feature
% input: y(i-1), yi x, i, o
% yi: label at i
% x: input sequence
% i: input position
% o: off set
%
% output:
% transition feature (k*1)
%
% Chang Gong
% 2014-11-29

function f = fTransFeatureTest(y1, y2, x, i, o)
k = 2;
f = zeros(k,1);

% feature 1, 3mer
x3_1 = mean(x(o+i-3:o+i-1))>0.5;
x3_2 = mean(x(o+i:o+i+2))>0.5;

f(1) = (x3_1==y1)&&(x3_2==y2);

% feature 2, 5mer
x5_1 = mean(x(o+i-5:o+i-1))>0.5;
x5_2 = mean(x(o+i:o+i+4))>0.5;

f(2) = (x5_1==y1)&&(x5_2==y2);
       
end