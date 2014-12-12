% Viterbi decoding
% input:
% x: feature vector to decode (n by f, n: sequence length; f: number of features)
% w: learned feature weight (m by k, m: posible discrete values of each feature combined; k: number of labels)
% v_start: learned starting weight (k by one)
% v_end: learned ending weight (k by one)
% v: learned transition weight (k by k)
% featureStart: for indexing in w (one by f+1)
% 
% output:
% y_est: labels assigned by Viterbi decoding
% p: log unnormalized potential.
%
% Chang Gong
% 2014-12-06

function [y_est, p]= viterbiDecoding(x,w,v_start,v_end,v,featureStart)
    
    % number of possible labels.
    nrYState = length(v_start);
    
    % length of sentence to decode
    seqLength = size(x,1);
    
    % number of X features for each node
    nrXfeature = size(featureStart, 2) - 1;
    
    %score
    S = zeros(nrYState,seqLength);
    %trace
    T = zeros(nrYState,seqLength);
    
    for i=1:seqLength
        if i==1
            %transition feature for first node: v_start 
            for j = 1:nrYState
                temps = 0;
                for f = 1:nrXfeature
                    if(x(i,f) ~= 0)
                       indw = featureStart(f) - 1 + x(i,f);
                        temps = temps + w(indw,j);
                    end
                end
                S(j,i) = v_start(j) + temps;
                T(j,i) = -1;
            end
        else
           for j = 1:nrYState % current
                trans = zeros(nrYState,1);
                for k = 1:nrYState % previous
                    trans(k) = S(k,i-1) + v(k,j);
                end
                [maxTrans, indTrans] = max(trans);
                
                temps = 0;
                for f = 1:nrXfeature
                    if(x(i,f) ~= 0)
                       indw = featureStart(f) - 1 + x(i,f);
                        temps = temps + w(indw,j);
                    end
                end
                S(j,i) = maxTrans + temps;
                T(j,i) = indTrans;
            end 
        end
        
        % v_last
        if i==seqLength
            S(:, i) = S(:, i) + v_end;
        end
    end
    
    % retrace
    ind_est = ones(seqLength,1)*-1;
    y_est = ones(seqLength,1)*-1;

    for i=seqLength:-1:1
        if i == seqLength
            [p,indS] = max(S(:,seqLength));
            ind_est(i) = indS;
        else
            ind_est(i) = T(ind_est(i+1),i+1);
        end
        y_est(i) = ind_est(i);
    end
    
end