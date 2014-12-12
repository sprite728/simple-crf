function [err, prec, recall] = crfChain_error_2(w,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences,type,useMex)

wv = [w(:);v_start(:);v_end(:);v(:)];

nSentences = size(sentences,1);

err = 0;
Z = 0;

y_true_all = zeros(0);
y_pred_all = zeros(0);

for s = 1:nSentences
    y_s = y(sentences(s,1):sentences(s,2));
    if useMex
        [nodePot,edgePot] = crfChain_makePotentialsC(X,wv,nFeatures,featureStart,sentences,s,nStates);
    else
    [nodePot,edgePot]=crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,s);
    end
    if strcmp(type,'infer')
        if useMex
        [nodeBel,edgeBel,logZ] = crfChain_inferC(nodePot,edgePot);
        else
        [nodeBel,edgeBel,logZ] = crfChain_infer(nodePot,edgePot);
        end
        [junk yhat] = max(nodeBel,[],2);
    else
        yhat = crfChain_decode(nodePot,edgePot);
    end
    
    y_true_all = [y_true_all; y_s];
    y_pred_all = [y_pred_all; yhat];
    
    err = err+sum(yhat~=y_s);
    Z = Z+length(y_s);
end

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

err=err/Z;