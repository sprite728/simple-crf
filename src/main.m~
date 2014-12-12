clear all;
%% read data

% experiment_activity;
experiment_ancestry;
% experiment_transportation;

nWords = size(X,1);
nStates = max(y);
nFeatures = max(X);

%% Preprocess sentences 
sentences = zeros(0,2);
j = 1;
for i = 1:length(y)
    if (i==1 || y(i-1) == 0) && y(i) ~= 0
        sentences(j,1) = i;
    end
    if (i==nWords || y(i+1) == 0) && y(i) ~= 0
        sentences(j,2) = i;
        j = j + 1;
    end
end

featureStart = cumsum([1 nFeatures(1:end)]); % data structure which relates high-level 'features' to elements of w
nSentences = size(sentences,1);

%% Set up training/testing indices
trainNdx = 1:floor(nSentences/2);
testNdx = floor(nSentences/2)+1:nSentences;

%% Parameter init
v_start = rand(nStates,1); % potential for tags to start sentences
v = rand(nStates,nStates); % potentials for transitions between tags
v_end = rand(nStates,1); % potential for tags to end sentences
w = rand(sum(nFeatures)*nStates,1); % potential of tag given individual features
wv = [w(:);v_start(:);v_end(:);v(:)];


%% Training on training set
% crfChain's minFunc is used, this contains the quasi-newton algorithm. 
[wv] = minFunc(@crfChain_loss,wv,[],X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:));


% Splitting 
nFeaturesTotal = featureStart(end)-1;
w = reshape(wv(1:nFeaturesTotal*nStates),nFeaturesTotal,nStates);
v_start = wv(nFeaturesTotal*nStates+1:nFeaturesTotal*nStates+nStates);
v_end = wv(nFeaturesTotal*nStates+nStates+1:nFeaturesTotal*nStates+2*nStates);
v = reshape(wv(nFeaturesTotal*nStates+2*nStates+1:end),nStates,nStates);

%% Inference & evaluation on the test set

nTestSentences = size(testNdx,2);

err = 0;
Z = 0;

y_true_all = zeros(0);
y_pred_all = zeros(0);

testSentences =  sentences(testNdx,:);

for s = 1:nTestSentences
    y_s = y(testSentences(s,1):testSentences(s,2));
    X_s = X(testSentences(s,1):testSentences(s,2),:);
%     if strcmp(type,'infer')
%         if useMex
%         [nodeBel,edgeBel,logZ] = crfChain_inferC(nodePot,edgePot);
%         else
%         [nodeBel,edgeBel,logZ] = crfChain_infer(nodePot,edgePot);
%         end
%         [junk yhat] = max(nodeBel,[],2);
%     else
%         yhat = crfChain_decode(nodePot,edgePot);
%     end

    yhat = viterbiDecoding(X_s, w, v_start, v_end, v, featureStart);

    y_true_all = [y_true_all; y_s];
    y_pred_all = [y_pred_all; yhat];
    
    err = err+sum(yhat~=y_s);
    Z = Z+length(y_s);
end

%% 
prec = zeros(max(y), 1);
recall = zeros(max(y), 1);

for cls = 1:max(y)
    TP = sum( (y_true_all==cls) & (y_pred_all == cls) );
    FN = sum( (y_true_all==cls) & (y_pred_all ~= cls) );
    FP = sum( (y_true_all~=cls) & (y_pred_all == cls) );
    TN = sum( (y_true_all~=cls) & (y_pred_all ~= cls) );
    
    prec(cls) = TP / (TP + FP);
    recall(cls) = TP / (TP + FN);
    
%     [prec(cls), tpr(cls), fpr(cls), thresh] = prec_rec(y_true_all==cls, y_pred_all==cls);
end

cls_weight = zeros(max(y), 1);

for i = 1: max(y)
    cls_weight(i) = sum(y==i);
end

err=err/Z
prec
recall
weighted_prec  = sum(prec .* cls_weight) / length(y)
weighted_recall = sum(recall .* cls_weight) / length(y)


