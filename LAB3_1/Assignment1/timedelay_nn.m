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
epochs = 100:100:300;

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
    
    % computing immse on TR and VAL
    Y_tr_pred = idnn(delayedInput_tr, initialInput_tr);
    error_tr = immse(cell2mat(delayedTarget_tr), cell2mat(Y_tr_pred));
    fprintf('Error on training set: %.5f\n', error_tr);
    
    Y_val_pred = idnn(delayedInput_val, initialInput_val);
    error_val = immse(cell2mat(delayedTarget_val), cell2mat(Y_val_pred));
    fprintf('Error on validation set: %.5f\n\n', error_val);
    
    if error_val < min_err_val
        min_err_val = error_val;
        best_h = h;
        best_l = l;
        best_lr = lr;
        %best_f = f;
        best_e = e;
    end
end

% model assessment
idnn = timedelaynet(1:best_l,best_h,'traingdx');
idnn.trainParam.lr = best_lr;
idnn.trainParam.epochs = best_e;
idnn.performParam.regularization = 0.1; % weight decay regularization
idnn.divideFcn = 'dividetrain';    

[delayedInput_tr,initialInput_tr,initialStates,delayedTarget] = preparets(idnn,X_train,Y_train);
[idnn, tr] = train(idnn,delayedInput_tr,delayedTarget,initialInput_tr,initialStates,'UseParallel','yes');
view(idnn)

% Y_test_pred = idnn(delayedInput_tr, initialInput_tr);
% error_test = immse(cell2mat(delayedTarget_val), cell2mat(Y_test_pred));
% fprintf('Error on test set: %.5f', error_test);
