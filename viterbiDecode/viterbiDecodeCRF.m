% Viterbi decoding
% input:
% fh: function handle for transition feature
% gh: function handle for state feature
% lambda: weight vectore for transition features (one*k)
% mu: weight vectore for state features (one*l)
% x: input sequence
% Y: set of labels
% seqLength: length of sequence to be decoded
% offset: start decoding from offset+1 in x.
%
% output:
% y_est: labels assigned by Viterbi decoding
% p: value.
%
% Chang Gong
% 2014-11-29

function [y_est, p] = viterbiDecodeCRF(fh, gh, lambda, mu, x, Y, seqLength, offset)

nrYState = size(Y,1);
%score
S = zeros(nrYState,seqLength);
%trace
T = zeros(nrYState,seqLength);

for i=1:seqLength
    if i==1
        % no transition feature for first node
        for j = 1:nrYState
            S(j,i) = mu*gh(Y(j),x,i,offset);
            T(j,i) = -1;
        end
    else
        for j = 1:nrYState % current
            trans = zeros(nrYState,1);
            for k = 1:nrYState % previous
                trans(k) = S(k,i-1) + lambda*fh(Y(k),Y(j),x,i,offset);
            end
            [maxTrans, indTrans] = max(trans);
            S(j,i) = maxTrans + mu*gh(Y(j),x,i,offset);
            T(j,i) = indTrans;
        end
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
    y_est(i) = Y(ind_est(i));
end

end