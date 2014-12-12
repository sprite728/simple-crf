load transportation_training.mat
X_train = X';

load transportation_testing.mat
X_test = X';

X_train = [X_train; X_test];

%%
y = X_train(:, 1);

[n_data, n_features] = size(X_train);
% n_features = n_features; % label is not a feature, thus remove it

X = X_train(:, 2:n_features);

% nWords = size(X,1);
% nStates = max(y);
% nFeatures = max(X);
% 
% %% Set up training/testing indices
% trainNdx = 1:floor(nSentences/2);
% testNdx = floor(nSentences/2)+1:nSentences;
% 
% %% Example of making potentials and doing inference with first sentence
% 
% s = 1;
% if useMex
%     [nodePot,edgePot] = crfChain_makePotentialsC(X,wv,nFeatures,featureStart,sentences,s,nStates);
%     [nodeBel,edgeBel,logZ] = crfChain_inferC(nodePot,edgePot);
% else
%     [nodePot,edgePot]=crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,s);
%     [nodeBel,edgeBel,logZ] = crfChain_infer(nodePot,edgePot);
% end
% 
% %% Compute Errors with random parameters
% 
% fprintf('Errors based on most likely sequence with random parameters:\n');
% 
% [trainErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:),'decode',useMex)
% [testErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(testNdx,:),'decode',useMex)
% 
% fprintf('Errors based on max marginals with random parameters:\n');
% 
% [trainErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:),'infer',useMex)
% [testErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(testNdx,:),'infer',useMex)
% 
% 
% %% Training
% 
% % Compute objective function over training data
% if useMex
%     maxSentenceLength = 1+max(sentences(:,2)-sentences(:,1));
%     crfChain_lossC2(wv,X,y,nStates,nFeatures,featureStart,sentences,maxSentenceLength);
% else
%     crfChain_loss(wv,X,y,nStates,nFeatures,featureStart,sentences);
% end
% 
% % Optimize parameters
% fprintf('Training...\n');
% if useMex
%     [wv] = minFunc(@crfChain_lossC2,[w(:);v_start;v_end;v(:)],[],X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:),maxSentenceLength);
% else
%     [wv] = minFunc(@crfChain_loss,wv,[],X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:));
% end
% 
% % Split up weights
% [w,v_start,v_end,v] = crfChain_splitWeights(wv,featureStart,nStates);
% 
% %% Decode/Infer/Sample based on first test example
% 
% s = testNdx(1);
% fprintf('True Labels for first test sentence:\n');
% y(sentences(s,1):sentences(s,2))'
% 
% fprintf('Most likely sequence under learned model for first test sentence:\n');
% if useMex
%     [nodePot,edgePot] = crfChain_makePotentialsC(X,wv,nFeatures,featureStart,sentences,s,nStates);
% else
%     [nodePot,edgePot]=crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,s);
% end
% yViterbi = crfChain_decode(nodePot,edgePot)'
% 
% fprintf('Sequence of marginally most likely states under learned model for first test sentence:\n');
% if useMex
%     nodeBel = crfChain_inferC(nodePot,edgePot);
% else
%     nodeBel = crfChain_infer(nodePot,edgePot);
% end
% [junk yMaxMarginal] = max(nodeBel,[],2);
% yMaxMarginal'
% 
% fprintf('Samples from model conditioned on features for first test sentence:\n');
% samples = crfChain_sample(nodePot,edgePot,10)'
% 
% %% Compute errors with learned parameters
% 
% cls_weight = zeros(max(y), 1);
% 
% for i = 1: max(y)
%     cls_weight(i) = sum(y==i);
% end
% 
% fprintf('Errors based on most likely sequence with learned parameters:\n');
% 
% [trainErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:),'decode',useMex)
% 
% [testErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(testNdx,:),'decode',useMex)
% weighted_prec  = sum(prec .* cls_weight) / length(y)
% weighted_recall = sum(recall .* cls_weight) / length(y)
% 
% fprintf('Errors based on max marginals with learned parameters:\n');
% 
% [trainErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(trainNdx,:),'infer',useMex)
% [testErr, prec, recall] = crfChain_error(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences(testNdx,:),'infer',useMex)
% weighted_prec  = sum(prec .* cls_weight) / length(y)
% weighted_recall = sum(recall .* cls_weight) / length(y)

