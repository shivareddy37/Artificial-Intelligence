%%************** Artificail Intelligence HW3*********************
%% Author Koyya, Shiva Karthik Reddy

%% LOADING THE GIVEN DATA SET
data=load('C:\Users\shiva\Documents\MATLAB\Pima_dataset\pima.txt');
[m,n]=size(data);

X=data(:,1:n-1);
Y=data(:,n);
%% normalizing the data
NDATA=zeros(size(X));
% calling normalization function
NDATA=normalization(X);
figure (1)
boxplot(X);
title('unnormalised data')
figure(2)
boxplot(NDATA);
title('normalised data') 

%% dividing the data for tranning crossvalidation and testing 
%calling fuction which performs the N-fold crossvalidation 
% in our case its 3 fold crossvalidation

indices=divideset(NDATA);
test=(indices==1);
TraiN=~test;

%% diving normalised dataset based on crossvalidatin result

TrainX=NDATA(TraiN==1,1:(n-1));
TrainY=Y(TraiN==1);
TestX=NDATA(test==1,1:(n-1));
TestY=Y(test==1);


% let 1 represent dibatic 
% let 0 represent non dibatic 
no_of_dibatic=sum(Y);
%% Machine learning algorithms

%% bosting(decesion tree)
error1=zeros(1,5);
no_of_lerner_cycles=[100 200 300 400 500];
tStart1=tic;
for i=1:5
ens = fitensemble(TrainX, TrainY, 'GentleBoost', no_of_lerner_cycles(i), 'Tree');
Ypredict = ens.predict(TestX);
error1(i) = sum(Ypredict~=TestY);
end
tElapsed1=toc(tStart1);
C1=confusionmat(TestY',Ypredict');
[M1 I1]=min(error1);
bestaccuracy1=((length(TestY)-M1)/(length(TestY)))*100;
best_para1=no_of_lerner_cycles(I1);
no_of_dibatic_pred1=sum(Ypredict);

s1=TestY+Ypredict;
 count=0;
for i=1:length(TestY)
    
    if(s1(i)==2)
        count=count+1;
    end
    
end
precesion1=count/no_of_dibatic_pred1;
    recall1=count/no_of_dibatic;


%% neural network(multi-layer feed forward)

% Neural net example
error2=zeros(1,5);
no_of_neurons=[10 20 30 40 50];
tStart2=tic;
for i=1:5
net = feedforwardnet(no_of_neurons(i), 'trainlm');
net.trainParam.showWindow = false; %disables display
net = train(net, TrainX', TrainY');
Ypredict = net(TestX');
Ypredict = round(Ypredict');% fix net output format
error2(i) = sum(Ypredict~=TestY);
end
tElapsed2=toc(tStart2);
C2=confusionmat(TestY',Ypredict');
[M2 I2]=min(error2);
bestaccuracy2=((length(TestY)-M2)/(length(TestY)))*100;
best_para2=no_of_neurons(I2);
no_of_dibatic_pred2=sum(Ypredict);
s2=TestY+Ypredict;
count=0;
for i=1:length(TestY)
    
    if(s2(i)==2)
        count=count+1;
    end
   
end
 precesion2=count/no_of_dibatic_pred2;
    recall2=count/no_of_dibatic;

%% SVM 
sigma=[1 2 3 4 5];
error3=zeros(1,5);
tStart3=tic;
for i=1:5
svmStruct = svmtrain(TrainX, TrainY, 'kernel_Function', 'rbf', 'rbf_sigma', sigma(i));
Ypredict = svmclassify(svmStruct, TestX);
error3(i) = sum(Ypredict~=TestY);
end
tElapsed3=toc(tStart3);
C3=confusionmat(TestY',Ypredict');
[M3 I3]=min(error3);
bestaccuracy3=((length(TestY)-M3)/(length(TestY)))*100;
best_para3=sigma(I3);
no_of_dibatic_pred3=sum(Ypredict);

s3=TestY+Ypredict;
count=0;
for i=1:length(TestY)
    
    if(s3(i)==2)
        count=count+1;
    end
    
end
precesion3=count/no_of_dibatic_pred3;
    recall3=count/no_of_dibatic;
    
    %% result discussion
    % bosting
    
    disp('results for boosting')
    disp('confusion matrix')
    disp(C1)
    disp(' best accuracy of classifier')
    disp(bestaccuracy1)
    disp('parameter for best accuracy')
    disp(best_para1)
    disp('precesion')
    disp(precesion1)
    disp('recall')
    disp(recall1)
    disp('time the classifer took in seconds')
    disp(tElapsed1)
    %neural
     disp('results for neural')
    disp('confusion matrix')
    disp(C2)
    disp(' best accuracy of classifier')
    disp(bestaccuracy2)
    disp('parameter for best accuracy')
    disp(best_para2)
    disp('precesion')
    disp(precesion2)
    disp('recall')
    disp(recall2)
    disp('time the classifer took in seconds')
    disp(tElapsed2)
    % svm
     disp('results for svm')
    disp('confusion matrix')
    disp(C3)
    disp(' best accuracy of classifier')
    disp(bestaccuracy3)
    disp('parameter for best accuracy')
    disp(best_para3)
    disp('precesion')
    disp(precesion3)
    disp('recall')
    disp(recall3)
    disp('time the classifer took in seconds')
    disp(tElapsed3)
    
    