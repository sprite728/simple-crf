function [nodePot,edgePot] = crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,s);
% according to the potential parameters w, v, v_start, v_end to calculate
% the potentials of every node and edge in the sentence

nNodes = sentences(s,2) - sentences(s,1) + 1; % number of nodes
edgePot = exp(v); % edgePot is a matrix nStates by nStates
nStates = length(v_start);
nodePot = zeros(nNodes,nStates); %initialize nodePot

featureMatr = X(sentences(s,1):sentences(s,2),:);

% w is a vector with X*nStates elements X is the total number of binary
% features e.g. a feature vector with three features, first feature has 3
% values, second 2 values, third 4 values, so it totally has 9 binary
% featues

for i=1:nNodes;
% Calculate the potential for every node
    features = featureMatr(i,:);
    for state=1:nStates;
        totPot = 0;
        for j = 1:nFeatures;
            if features(j)~=0;
                BinaryFeatureIndex = (state - 1 ) * ( featureStart(end) -1 ) + featureStart(j) +features(j) -1;
                totPot = totPot + w(BinaryFeatureIndex);
            end;
        end;
        nodePot(i,state) = totPot;
    end;
    
end;

% Modify the Potential of start node and end node

nodePot(1,:) =nodePot(1,:) + v_start';
nodePot(nNodes,:) = nodePot(nNodes,:) + v_end';
nodePot = exp(nodePot);
