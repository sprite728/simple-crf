load activity_discretized.mat

data = data';
y = data(:, 1);
X = data(:, 2:16); % only take the first 20 features for testing

