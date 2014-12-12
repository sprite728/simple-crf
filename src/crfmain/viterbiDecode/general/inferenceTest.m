%% setup
offset = 4;
seqLength = 10;

Y = [0,1]';

%% parameters

lambda = [1.1,2.3];
mu = [0.7,1.3,2.3];

minInputLength = seqLength+offset*2;
x = rand(1,minInputLength)>0.5;


%% inference : Viterbi decoding

[y_est,p] = viterbiDecodeCRF(@fTransFeatureTest, @fStateFeatureTest, lambda, mu, x, Y, seqLength, offset);


disp(y_est');

%% check
all_Y = (dec2bin(0:(2^seqLength)-1)=='1') ;
nrCase = size(all_Y,1);
perm_score = zeros(nrCase,1);

for n=1:nrCase
    y = all_Y(n,:);
    E = 0;
    for i=1:seqLength
        if i > 1
            E = E + lambda*fTransFeatureTest(y(i-1),y(i),x,i,offset) + mu*fStateFeatureTest(y(i),x,i,offset);
        else
            E = E + mu*fStateFeatureTest(y(i),x,i,offset);
        end
    end
    perm_score(n) = E;
end

disp(all_Y(perm_score==max(perm_score),:));

