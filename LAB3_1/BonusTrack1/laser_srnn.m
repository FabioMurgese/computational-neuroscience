%%%%%%%%%%%%%%% Recurrent Neural Network %%%%%%%%%%%%%%%
clear variables;

load laser_dataset; % import data
allData = cell2mat(laserTargets);
allData = rescale(allData,-1,1);
inputData = allData(1:end-1);
input = num2cell(inputData);
targetData = allData(2:end);
target = num2cell(targetData);

steps = 5000;
steps_test = length(inputData)-steps;
val_steps = 4000;

% design set
X_design = input(1:steps);
Y_design = target(1:steps);
% test set
X_test = input(steps+1:end);
Y_test = target(steps+1:end);
% training set
X_train = X_design(1:val_steps);
Y_train = Y_design(1:val_steps);
% validation set
X_val = X_design(val_steps+1:end);
Y_val = Y_design(val_steps+1:end);


% parameters for grid search
hiddenSizes = 10:30:100;
lrs = [0.0001 0.001 0.01 0.1];
%functions = ['traingdm' 'traingdx' 'trainrp'];
epochs = 100:300:1000;

[H, LR, E] = ndgrid(hiddenSizes,lrs,epochs);
grid = [H(:) LR(:) E(:)];

min_err_val = inf;

for g = 1:size(grid,1)
    
    h = grid(g,1);
    lr = grid(g,2);
    e = grid(g,3);
    fprintf('Hidden size: %d - Learning rate: %.4f - Epochs: %d\n', h, lr, e);
    
    srnn = layrecnet(1,h,'traingdx');
    srnn.trainParam.lr = lr;
    srnn.trainParam.epochs = e;
    srnn.performParam.regularization = 0.1; % weight decay regularization
    srnn.divideFcn = 'dividetrain';
    srnn.trainParam.showWindow = false;
    
    % prepare timeseries
    [delayedInput_tr,initialInput_tr,initialStates_tr,delayedTarget_tr] = preparets(srnn,X_train,Y_train);  % TR
    [delayedInput_val,initialInput_val,initialStates_val,delayedTarget_val] = preparets(srnn,X_val,Y_val);  % VAL
    
    % train on TR
    [srnn, tr] = train(srnn,delayedInput_tr,delayedTarget_tr,initialInput_tr,initialStates_tr,'UseParallel','yes');
    plot(tr.perf);
    
    % computing immse on TR and VAL
    Y_tr_pred = srnn(X_train);
    error_tr = immse(cell2mat(Y_train), cell2mat(Y_tr_pred));
    fprintf('Error on training set: %.5f\n', error_tr);
    
    Y_val_pred = srnn(X_val);
    error_val = immse(cell2mat(Y_val), cell2mat(Y_val_pred));
    fprintf('Error on validation set: %.5f\n\n', error_val);
    
    if error_val < min_err_val
        min_err_val = error_val;
        min_err_tr = error_tr;
        best_h = h;
        best_lr = lr;
        best_e = e;
        
        best_error_val = error_val;
    end
end

% model assessment
fprintf('Best configuration:\nHidden size: %d - Learning rate: %.4f - Epochs: %d\n', best_h, best_lr, best_e);
srnn = layrecnet(1,best_h,'traingdx');
srnn.trainParam.lr = best_lr;
srnn.trainParam.epochs = best_e;
srnn.performParam.regularization = 0.1; % weight decay regularization
srnn.divideFcn = 'dividetrain';    

[delayedInput,initialInput,initialStates,delayedTarget] = preparets(srnn,X_design,Y_design);
[srnn, tr] = train(srnn,delayedInput,delayedTarget,initialInput,initialStates,'UseParallel','yes');
view(srnn)

% computing immse on TR (design) and TEST
Y_tr_pred = srnn(X_design);
error_tr = immse(cell2mat(Y_design), cell2mat(Y_tr_pred));
fprintf('Error on training (design) set: %.5f\n', error_tr);

Y_test_pred = srnn(X_test);
error_test = immse(cell2mat(Y_test), cell2mat(Y_test_pred));
fprintf('Error on test set: %.5f\n', error_test);

fig = figure;
plot(tr.perf);
xlabel('epochs')
ylabel('error')
title('Learning curve TR (design set)');
print(fig,'srnn/srnn_laser_learning_curve.png','-dpng')


time = 1:steps;
time_test = 1:steps_test;
sz = 25;
fig = figure;
tiledlayout(2,1)
% Top plot
nexttile
hold on;
plot(time,cell2mat(Y_design));  % ground truth
plot(time,cell2mat(Y_tr_pred)); % predictions
% scatter(time,cell2mat(Y_design),sz);  % ground truth
% scatter(time,cell2mat(Y_tr_pred),sz); % predictions
hold off;
legend('target','predictions');
title('TR+VAL target and output signals');

% Bottom plot
nexttile
hold on;
plot(time_test,cell2mat(Y_test));  % ground truth
plot(time_test,cell2mat(Y_test_pred)); % predictions
% scatter(time_test,cell2mat(Y_test),sz);  % ground truth
% scatter(time_test,cell2mat(Y_test_pred),sz); % predictions
hold off;
legend('target','predictions');
title('TEST target and output signals');
print(fig,'srnn/srnn_laser_target_predictions.png','-dpng')