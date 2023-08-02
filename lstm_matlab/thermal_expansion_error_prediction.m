%	Date : 2023.07.25
%	Programmer : Ruoheng Du
%	Description : Thermal Expansion Error Prediction

%% 

%   load data
data = readtable('thermal_expansion_error.xlsx');


%% 

%   90% : Training Dataset
%   10% : Test Dataset
numTimeStepsTrain = round(0.9*size(data));
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);


%% 

%   Split into XTrain & YTrain
XTrain = dataTrain(:, 1:end-1);
YTrain = dataTrain(:, end);


%% 

%   Normalize Training Data Set
XTrain_mu = mean([XTrain{:,:}],1);
XTrain_sig = std([XTrain{:,:}],0,1);
YTrain_mu = mean([YTrain{:,:}],1);
YTrain_sig = std([YTrain{:,:}],0,1);

for i = 1:size(XTrain,2)
    for j = 1:size(XTrain,1)
        XTrain(j,i) = (XTrain(j,i) - XTrain_mu(1,i)) ./ XTrain_sig(1,i) ;
        YTrain(j,1) = (YTrain(j,1) - YTrain_mu(1,1)) ./ YTrain_sig(1,1) ;
    end
end


%% 

%   Define LSTM
numFeatures = size(XTrain{1,:},2);
numResponses = size(YTrain{1,:},1);
numHiddenUnits = 64;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 5000;
miniBatchSize = 512;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);
	
%	Train LSTM
net = trainNetwork(XTrain,YTrain,layers,options);


%% 

%	Normalize Testing Data Set
XTest_mu = mean([XTest{:,:}],1);
XTest_sig = std([XTest{:,:}],0,1);
YTest_mu = mean([YTest{:,:}],1);
YTest_sig = std([YTest{:,:}],0,1);

for i = 1:size(XTest,2)
    for j = 1:size(XTest,1)
        XTest(j,i) = (XTest(j,i) - XTest_mu(1,i)) ./ XTest_sig(1,i) ;
        YTest(j,i) = (YTest(j,1) - YTest_mu(1,1)) ./ YTest_sig(1,1) ;
    end
end

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));


%% 

%   Predict
numTimeStepsTest = size(XTest,2);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%   RMSE calculation of test data set
YTest = dataTest(2:end);
YTest = (YTest - mu) / sig;
rmse = sqrt(mean((YPred-YTest).^2));

%	Denormalize Data
YPred = sig*YPred + mu;
YTest = sig*YTest + mu;

%   X Label : Collect Day
x_data = datetime(data.collect_day);
x_train = x_data(1:numTimeStepsTrain+1);
x_train = x_train';
x_pred = x_data(numTimeStepsTrain:numTimeStepsTrain+numTimeStepsTest);

%   Train + Predict Plot
figure
plot(x_train(1:end-1),dataTrain(1:end-1))
hold on
plot(x_pred,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time")
ylabel("Thermal Expansion Error")
title("Train vs Predict")
legend(["Observed" "Forecast"])

%  RMSE Plot : Test + Predict Plot
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Thermal Expansion Error")
title("Test vs Predict")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time")
ylabel("Prediction Error")
title("RMSE = " + rmse)

%   Train + Test + Predict Plot
figure
plot(x_data,Y)
hold on
plot(x_pred,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time")
ylabel("Thermal Expansion Error")
title("Compare Data")
legend(["Raw" "Forecast"])