%%%%%%%%%%%%%%% Time Delay Neural Network %%%%%%%%%%%%%%%
clear variables;

load('NARMA10timeseries.mat');  % import data
input = NARMA10timeseries.input;
target = NARMA10timeseries.target;

steps = 5000;
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

% min_err_tr = inf;
min_err_val = inf;

for g = 1:size(grid,1)
    
    h = grid(g,1);
    l = grid(g,2);
    lr = grid(g,3);
    e = grid(g,4);
    fprintf('Hidden size: %d - Window length: %d - Learning rate: %.4f - Epochs: %d\n', h, l, lr, e);
    
    idnn = timedelaynet(1:l,h,'traingdx');
    idnn.trainParam.lr = lr;
    idnn.trainParam.epochs = e;
    idnn.performParam.regularization = 0.1; % weight decay regularization
    idnn.divideFcn = 'dividetrain';  
    
    % prepare timeseries
    [delayedInput_tr,initialInput_tr,initialStates_tr,delayedTarget_tr] = preparets(idnn,X_train,Y_train);  % TR
    [delayedInput_val,initialInput_val,initialStates_val,delayedTarget_val] = preparets(idnn,X_val,Y_val);  % VAL
    
    % train on TR
    [idnn, tr] = train(idnn,delayedInput_tr,delayedTarget_tr,initialInput_tr,initialStates_tr,'UseParallel','yes');
    plot(tr.perf);
    
    % computing immse on TR and VAL
    Y_tr_pred = idnn(X_train);
    error_tr = immse(cell2mat(Y_train), cell2mat(Y_tr_pred));
    fprintf('Error on training set: %.5f\n', error_tr);
    
    Y_val_pred = idnn(X_val);
    error_val = immse(cell2mat(Y_val), cell2mat(Y_val_pred));
    fprintf('Error on validation set: %.5f\n\n', error_val);
    
    if error_val < min_err_val
        min_err_val = error_val;
        best_h = h;
        best_l = l;
        best_lr = lr;
        %best_f = f;
        best_e = e;
        
        best_error_tr = error_tr;
        best_error_val = error_val;
    end
end

% model assessment
fprintf('Best configuration:\nHidden size: %d - Window length: %d - Learning rate: %.4f - Epochs: %d\n', best_h, best_l, best_lr, best_e);
idnn = timedelaynet(1:best_l,best_h,'traingdx');
idnn.trainParam.lr = best_lr;
idnn.trainParam.epochs = best_e;
idnn.performParam.regularization = 0.1; % weight decay regularization
idnn.divideFcn = 'dividetrain';    

[delayedInput,initialInput,initialStates,delayedTarget] = preparets(idnn,X_design,Y_design);
[idnn, tr] = train(idnn,delayedInput,delayedTarget,initialInput,initialStates,'UseParallel','yes');
view(idnn)

% computing immse on TR (design) and TEST
Y_tr_pred = idnn(X_design);
error_tr = immse(cell2mat(Y_design), cell2mat(Y_tr_pred));
fprintf('Error on training (design) set: %.5f\n', error_tr);

Y_test_pred = idnn(X_test);
error_test = immse(cell2mat(Y_test), cell2mat(Y_test_pred));
fprintf('Error on test set: %.5f', error_test);

fig = figure;
plot(tr.perf);
xlabel('epochs')
ylabel('error')
title('Learning curve TR (design set)');
print(fig,'img/idnn_learning_curve.png','-dpng')


time = 1:steps;
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
plot(time,cell2mat(Y_test));  % ground truth
plot(time,cell2mat(Y_test_pred)); % predictions
% scatter(time,cell2mat(Y_test),sz);  % ground truth
% scatter(time,cell2mat(Y_test_pred),sz); % predictions
hold off;
legend('target','predictions');
title('TEST target and output signals');
print(fig,'img/idnn_target_predictions.png','-dpng')