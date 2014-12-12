function [negLogLoss,gradient] = crfChain_loss(wv,X,y,nStates,nFeatures,featureStart,sentences);
    nSentences = size(sentences,1);
    f = 0;  %Log loss function value
    %initialize the gradient
    totBinaryFeatures = featureStart(end)-1;
    w = reshape(wv(1:totBinaryFeatures*nStates),totBinaryFeatures,nStates);
    v_start = wv(totBinaryFeatures*nStates+1:totBinaryFeatures*nStates+nStates);
    v_end = wv(totBinaryFeatures*nStates+nStates+1:totBinaryFeatures*nStates+2*nStates);
    v = reshape(wv(totBinaryFeatures*nStates+2*nStates+1:end),nStates,nStates);
    gradientw = zeros(totBinaryFeatures,nStates);
    gradientv = zeros(nStates,nStates);
    gradientv_start = zeros(nStates,1);
    gradientv_end = zeros(nStates,1);
    for i=1:nSentences;
        %calculate the potential of nodes and edges
        [nodePot,edgePot] = crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,i);
        %calculate the belief of nodes and edges
        [nodeBel,edgeBel,logZ] = crfChain_infer(nodePot,edgePot); 
        %logZ can be used to calculate the value of loss function
        
        nodesNum = sentences(i,2) - sentences(i,1) + 1;
        y_true = y(sentences(i,1):sentences(i,2),:);
        %Calculate the log loss function
        for j=1:nodesNum;
            f = f + log(nodePot(j,y_true(j)));
        end;
        for j=1:nodesNum-1;
            f = f + log(edgePot(y_true(j),y_true(j+1)));
        end;
        f = f - logZ;
        
        % upgrade gradient
        % node gradient
        for j=1:nodesNum;
            features = X(sentences(i,1) + j -1,:);
            for f=1:nFeatures;
                if features(f)~=0;
                    binaryFeatureIndex = featureStart(f) + features(f) - 1;
                    for s=1:nStates;
                        if y_true(j)==s;
                            gradientw(binaryFeatureIndex,s) = gradientw(binaryFeatureIndex,s) - 1 + nodeBel(j,s);
                        else
                            gradientw(binaryFeatureIndex,s) = gradientw(binaryFeatureIndex,s) + nodeBel(j,s);
                        end;
                    end;
                end;
                
            end;
        end;
        % edge gradient
        for j=1:nodesNum-1;
            for s1=1:nStates;
                for s2=1:nStates;
                    if (y_true(j)==s1) && (y_true(j+1)==s2);
                        gradientv(s1,s2) = gradientv(s1,s2) - 1 + edgeBel(s1,s2,j);
                    else
                        gradientv(s1,s2) = gradientv(s1,s2) + edgeBel(s1,s2,j);
                    end;
                    
                end;
            end;
        end;
        % v_start and v_end gradient
        for s=1:nStates;
            if y_true(1)==s;
                gradientv_start(s,1) = gradientv_start(s,1) - 1 + nodeBel(1,s);
            else
                gradientv_start(s,1) = gradientv_start(s,1) + nodeBel(1,s);
            end;
            if y_true(nodesNum)==s;
                gradientv_end(s,1) = gradientv_end(s,1) -1 + nodeBel(nodesNum,s);
            else
                gradientv_end(s,1) = gradientv_end(s,1) -1 +nodeBel(nodesNum,s);
            end;
        end;
        
        
    end;
    
    drawnow;
    negLogLoss = -f;
    gradient = [gradientw(:);gradientv_start;gradientv_end;gradientv(:)];
    
    

