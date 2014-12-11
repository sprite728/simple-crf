load ancestry_hmm.mat

data = data';
true_state = true_state';

hmm.prior = [1 0];
hmm.transmat = [0.998, 0.002; 0.998, 0.002];

hmm.transmat = mk_stochastic(hmm.transmat);

hmm.obsmat = rand(2, 6);
hmm.obsmat = mk_stochastic(hmm.obsmat);

n_train = 60;
n_test = size(true_state, 1) - n_train;
n_total = size(true_state, 1);

[LL0, hmm.prior, hmm.transmat, hmm.obsmat] = dhmm_em(data(1:n_train,:), hmm.prior, hmm.transmat, hmm.obsmat);

% smoothing of HMM observation parameter: set floor value 1.0e-5
hmm.obsmat = max(hmm.obsmat, 1.0e-5);


d = size(true_state, 2);

% Map the class labels to 1 and 0 for easy error calculation
ind = find(true_state==2);
true_state(ind) = 0;

path = zeros(size(true_state));

for dt = n_train+1:n_total
    fprintf('index = %d\n', dt);
    hmm.B = mk_dhmm_obs_lik( data(dt,:), hmm.obsmat);
    path(dt,:) = viterbi_path(hmm.prior, hmm.transmat, hmm.B);
end

% Map the class labels to 1 and 0 for easy error calculation
ind = find(path==2);
path(ind) = 0;


err = abs(path(n_train+1:n_total,:) - true_state(n_train+1:n_total,:));
err = sum(sum(err))/(n_test*d);
