%%%%%%%%%%%%%%% Time Delay Neural Network - Laser task %%%%%%%%%%%%%%%
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
lengths = [2 3 4];
lrs = [0.0001 0.001 0.01 0.1];
%functions = ['traingdm' 'traingdx' 'trainrp'];
epochs = 100:300:1000;

[H, L, LR, E] = ndgrid(hiddenSizes,lengths,lrs,epochs);
grid = [H(:) L(:) LR(:) E(:)];

Ers_tr = [];
Ers_val = [];

for g = 1:size(grid,1)
    
    fprintf('\n#%d/%d: ', g, size(grid,1));
    h = grid(g,1);
    l = grid(g,2);
    lr = grid(g,3);
    e = grid(g,4);
    fprintf('Hidden size: %d - Window length: %d - Learning rate: %.4f - Epochs: %d\n',h,l,lr,e);
    
    idnn = timedelaynet(1:l,h,'traingdx');
    idnn.trainParam.lr = lr;
    idnn.trainParam.epochs = e;
    idnn.performParam.regularization = 0.1; % weight decay regularization
    idnn.divideFcn = 'dividetrain';
    idnn.trainParam.showWindow = false;
    
    % prepare timeseries
    [delayedInput_tr,initialInput_tr,initialStates_tr,delayedTarget_tr] = preparets(idnn,X_train,Y_train);  % TR
    [delayedInput_val,initialInput_val,initialStates_val,delayedTarget_val] = preparets(idnn,X_val,Y_val);  % VAL
    
    % train on TR
    [idnn, tr] = train(idnn,delayedInput_tr,delayedTarget_tr,initialInput_tr,initialStates_tr,'UseParallel','yes');
    plot(tr.perf);
    
    % computing immse on TR and VAL
    Y_tr_pred = idnn(X_train);
    error_tr = immse(cell2mat(Y_train), cell2mat(Y_tr_pred));
    Ers_tr(end+1) = error_tr;
    fprintf('Error on training set: %.5f\n', error_tr);
    
    Y_val_pred = idnn(X_val);
    error_val = immse(cell2mat(Y_val), cell2mat(Y_val_pred));
    Ers_val(end+1) = error_val;
    fprintf('Error on validation set: %.5f\n\n', error_val);
    
end

[val_mse, idx] = min(Ers_val);
tr_mse = Ers_tr(idx);
h = grid(idx,1);
l = grid(idx,2);
lr = grid(idx,3);
e = grid(idx,4);
fprintf('\nBest hyper-params:\nHidden size: %d - Window length: %d - Learning rate: %.4f - Epochs: %d\nTraining MSE: %.5f\nValidation MSE: %.5f\n',h,l,lr,e,tr_mse,val_mse);

% model assessment
idnn = timedelaynet(1:l,h,'traingdx');
idnn.trainParam.lr = lr;
idnn.trainParam.epochs = e;
idnn.performParam.regularization = 0.1; % weight decay regularization
idnn.divideFcn = 'dividetrain';    

[delayedInput,initialInput,initialStates,delayedTarget] = preparets(idnn,X_design,Y_design);
[idnn, tr] = train(idnn,delayedInput,delayedTarget,initialInput,initialStates,'UseParallel','yes');
view(idnn)

% computing immse on TR+VAL (design) and TEST
Y_tr_pred = idnn(X_design);
design_mse = immse(cell2mat(Y_design), cell2mat(Y_tr_pred));
fprintf('Design (TR+VAL) MSE: %.5f\n', design_mse);

Y_test_pred = idnn(X_test);
test_mse = immse(cell2mat(Y_test), cell2mat(Y_test_pred));
fprintf('Test MSE: %.5f\n', test_mse);

fig = figure;
plot(tr.perf);
xlabel('epochs')
ylabel('error')
title('Learning curve TR (design set)');
savefig('idnn/idnn_laser_learning_curve')
print(fig,'idnn/idnn_laser_learning_curve.png','-dpng')

save('idnn/outputs.mat','tr_mse','val_mse','design_mse','test_mse','idnn','tr')

fig = figure;
tiledlayout(2,1)
% Top plot
nexttile
hold on;
plot(1:size(cell2mat(Y_design),2),cell2mat(Y_design));  % ground truth
plot(1:size(cell2mat(Y_tr_pred),2),cell2mat(Y_tr_pred)); % predictions
hold off;
legend('target','predictions');
title('TR+VAL target and output signals');

% Bottom plot
nexttile
hold on;
plot(1:size(cell2mat(Y_test),2),cell2mat(Y_test));  % ground truth
plot(1:size(cell2mat(Y_test_pred),2),cell2mat(Y_test_pred)); % predictions
hold off;
legend('target','predictions');
title('TEST target and output signals');
savefig('idnn/idnn_laser_target_predictions')
print(fig,'idnn/idnn_laser_target_predictions.png','-dpng')