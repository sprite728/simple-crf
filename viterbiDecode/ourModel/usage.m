% Generate Synthetic Data

% Notes:
%   - X is categorical, each element X(i,j) contains the value of feature j for word i,
%       a value of '0' means ignore the feature for this training example
%   - y is cateogircal, each element y(i) contains the label for word i
%       a value of '0' indicates the position between sentences
[X,y] = crfChain_genSynthetic;

nWords = size(X,1);
nStates = max(y);
nFeatures = max(X);

%% Initialize parameters and data structures

[w,v_start,v_end,v] = crfChain_initWeights(nFeatures,nStates,'randn');
featureStart = cumsum([1 nFeatures(1:end)]); % data structure which relates high-level 'features' to elements of w
sentences = crfChain_initSentences(y);
nSentences = size(sentences,1);

wv = [w(:);v_start(:);v_end(:);v(:)];

%% Set up training/testing indices
trainNdx = 1:floor(nSentences/2);
testNdx = floor(nSentences/2)+1:nSentences;

%% Training

% Compute objective function over training data
crfChain_loss(wv,X,y,nStates,nFeatures,featureStart,sentences);


% Optimize parameters
fprintf('Training...\n');
[wv] = minFunc(@crfChain_loss,wv,[],X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:));

% Split up weights
[w,v_start,v_end,v] = crfChain_splitWeights(wv,featureStart,nStates);


%% vitervi decoding
nrTest = size(testNdx,2);

% first column: compare true label with 
% second column: compare online code deconding with our viterbi
error_viterbi = ones(nrTest,2)*-1;

i = 0;
for s = testNdx
    i = i + 1;
    % original
    [nodePot,edgePot]=crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,s);
    yViterbi = crfChain_decode(nodePot,edgePot);

    % adapt for our version
    testX = X(sentences(s,1):sentences(s,2),:);
    ourYviterbi = viterbiDecoding(testX, w,v_start,v_end,v,featureStart);
    
    % check labels
    error_viterbi(i, 1) = sum(y(sentences(s,1):sentences(s,2)) ~= yViterbi);
    error_viterbi(i, 2) = sum(yViterbi ~= ourYviterbi);
end
error_viterbi
