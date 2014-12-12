X1=csvread('/Users/jeff/Develop/simple-crf/experiments/ancestry_data_process/feature_window10000.txt');
y1=csvread('/Users/jeff/Develop/simple-crf/experiments/ancestry_data_process/True_label.txt');
X2=csvread('/Users/jeff/Develop/simple-crf/experiments/ancestry_data_process/feature_window10000_v2.txt');
y2=csvread('/Users/jeff/Develop/simple-crf/experiments/ancestry_data_process/True_label_v2.txt');

X=[X1;X2];
y=[y1;y2];
