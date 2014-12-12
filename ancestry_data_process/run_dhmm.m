O = 6;
Q = 2;

X1=csvread('/Users/songsy/Documents/github/simple-crf/ancestry_data_process/feature_window10000.txt');
y1=csvread('/Users/songsy/Documents/github/simple-crf/ancestry_data_process/True_label.txt');
X2=csvread('/Users/songsy/Documents/github/simple-crf/ancestry_data_process/feature_window10000_v2.txt');
y2=csvread('/Users/songsy/Documents/github/simple-crf/ancestry_data_process/True_label_v2.txt');

X=[X1;X2];
y=[y1;y2];

data1 = zeros(8955,120);
true_state1 = zeros(8955,120);
for i=1:120
    for j=1:8955
        index=(i-1)*8956+j;
        m=X1(index,:);
        true_state1(j,i)=y1(index);
        if m(1)==1 && m(2)==2 && m(3)==1
            data1(j,i)=1;
        elseif m(1)==1 && m(2)==2 && m(3)==2
            data1(j,i)=2;
        elseif m(1)==2 && m(2)==1 && m(3)==1
            data1(j,i)=3;
        elseif m(1)==2 && m(2)==1 && m(3)==2
            data1(j,i)=4;
        elseif m(1)==2 && m(2)==2 && m(3)==1
            data1(j,i)=5;
        elseif m(1)==2 && m(2)==2 && m(3)==2
            data1(j,i)=6;
        else
            m
        end
    end
end
data2 = zeros(9662,120);
true_state2 = zeros(9662,120);
for i=1:120
    for j=1:9662
        index=(i-1)*9663+j;
        m=X2(index,:);
        true_state2(j,i)=y2(index);
        if m(1)==1 && m(2)==2 && m(3)==1
            data2(j,i)=1;
        elseif m(1)==1 && m(2)==2 && m(3)==2
            data2(j,i)=2;
        elseif m(1)==2 && m(2)==1 && m(3)==1
            data2(j,i)=3;
        elseif m(1)==2 && m(2)==1 && m(3)==2
            data2(j,i)=4;
        elseif m(1)==2 && m(2)==2 && m(3)==1
            data2(j,i)=5;
        elseif m(1)==2 && m(2)==2 && m(3)==2
            data2(j,i)=6;
        else
            m
        end
    end
end


train_data = data1';
train_true = true_state1';
test_data = data2';
test_true = true_state2';

% calculating the transition and emmision probablity using trainning data
m=length(find(train_true==1))/size(train_true,1)/size(train_true,2);
prior0=[m,1-m];
transmat0=zeros(Q,Q);
obsmat0=zeros(Q,O);

for i=1:120
    for j=2:8955
        if train_true(i,j-1)==1 && train_true(i,j)==1
            transmat0(1,1)=transmat0(1,1)+1;
        elseif train_true(i,j-1)==1 && train_true(i,j)==2
            transmat0(1,2)=transmat0(1,2)+1;
        elseif train_true(i,j-1)==2 && train_true(i,j)==1
            transmat0(2,1)=transmat0(2,1)+1;
        elseif train_true(i,j-1)==2 && train_true(i,j)==2
            transmat0(2,2)=transmat0(2,2)+1;
        end
    end
end

for i=1:120
    for j=1:8955
        a=train_data(i,j);
        b=train_true(i,j);
        obsmat0(b,a)=obsmat0(b,a)+1;
    end
end
transmat0=transmat0+5000;  %add some pseudocount
transmat0(1,:)=transmat0(1,:)/sum(transmat0(1,:));
transmat0(2,:)=transmat0(2,:)/sum(transmat0(2,:));
obsmat0=obsmat0+500; %add some pseudocount
obsmat0(1,:)=obsmat0(1,:)/sum(obsmat0(1,:));
obsmat0(2,:)=obsmat0(2,:)/sum(obsmat0(2,:));

B=multinomial_prob(train_data,obsmat0);
[train_path0]=viterbi_path(prior0,transmat0,B);

B=multinomial_prob(test_data,obsmat0);
[test_path0]=viterbi_path(prior0,transmat0,B);

train_error=0;
test_error=0;
TP = 0;
FP = 0;
FN = 0;
for i=1:120
    for j=1:8955
        index=(i-1)*8955+j;
        if train_path0(index)~=train_true(i,j);
                train_error=train_error+1;
            if train_path0(index)==2
                FP = FP+1;
            else
                FN= FN+1;
            end
        else
            if train_path0(index)==2
                TP = TP+1;
            end
        end
    end
end
precision=TP/(TP+FP)
recall=TP/(TP+FN)
test_error=0;
TP = 0;
FP = 0;
FN = 0;
for i=1:120
    for j=1:9662
        index=(i-1)*9662+j;
        if test_path0(index)~=test_true(i,j);
            test_error=test_error+1;
            if test_path0(index)==2
                FP = FP+1;
            else
                FN= FN+1;
            end
         else
            if test_path0(index)==2
                TP = TP+1;
            end 
        end
    end
end
precision=TP/(TP+FP)
recall=TP/(TP+FN)
train_error=train_error/8955/120
test_error=test_error/9662/120


% Use Baum-welch

% initial guess of parameters
prior1 = prior0;
transmat1 = mk_stochastic(rand(Q,Q));
obsmat1 = mk_stochastic(rand(Q,O));

% improve guess of parameters using EM
[LL, prior2, transmat2, obsmat2] = dhmm_em(train_data, prior1, transmat1, obsmat1, 'max_iter', 100);
LL

% use model to compute log likelihood
loglik = dhmm_logprob(data, prior2, transmat2, obsmat2)
% log lik is slightly different than LL(end), since it is computed after the final M step

B=multinomial_prob(train_data,obsmat2);
[train_path]=viterbi_path(prior2,transmat2,B);

B=multinomial_prob(test_data,obsmat2);
[test_path]=viterbi_path(prior2,transmat2,B);

train_error=0;
test_error=0;
TP = 0;
FP = 0;
FN = 0;
for i=1:120
    for j=1:8955
        index=(i-1)*8955+j;
        if train_path(index)~=train_true(i,j);
                train_error=train_error+1;
            if train_path(index)==2
                FP = FP+1;
            else
                FN= FN+1;
            end
        else
            if train_path(index)==2
                TP = TP+1;
            end
        end
    end
end
precision=TP/(TP+FP)
recall=TP/(TP+FN)
test_error=0;
TP = 0;
FP = 0;
FN = 0;
for i=1:120
    for j=1:9662
        index=(i-1)*9662+j;
        if test_path(index)~=test_true(i,j);
            test_error=test_error+1;
            if test_path(index)==2
                FP = FP+1;
            else
                FN= FN+1;
            end
         else
            if test_path(index)==2
                TP = TP+1;
            end 
        end
    end
end
precision=TP/(TP+FP)
recall=TP/(TP+FN)
train_error=train_error/8955/120
test_error=test_error/9662/120



