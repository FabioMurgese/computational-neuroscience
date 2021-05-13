%%%%%%%%%%%%%%% Echo State Network %%%%%%%%%%%%%%%
clear variables;

load('NARMA10timeseries.mat');  % import data
input = cell2mat(NARMA10timeseries.input);
target = cell2mat(NARMA10timeseries.target);

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
input_scaling = [0.5 1 2];
Nrs = [5 10 25 50 100 200];  % reservoir dimension (number of recurrent units)
rho_values = [0.1 0.5 0.9 1.2 2];  % spectral radius
lambdas = [0.0001 0.001 0.01 0.1];  % readout regularization for ridge regression
% connectivity = [0.9 0.7 0.5];  % percentage of connectivity among reservoir units

[I, NR, R, L] = ndgrid(input_scaling,Nrs,rho_values,lambdas);
grid = [I(:) NR(:) R(:) L(:)];
% [I, NR, R, L, C] = ndgrid(input_scaling,Nrs,rho_values,lambdas,connectivity);
% grid = [I(:) NR(:) R(:) L(:) C(:)];

Ers_tr = [];
Ers_val = [];

for g = 1:size(grid,1)
    
    omega_in = grid(g,1);
    Nr = grid(g,2);
    rho = grid(g,3);
    l = grid(g,4);
%     c = grid(g,5);
    
    guesses = 10;  % network guesses for each reservoir hyper-parametrization
    Nu = size(X_train,1);
    trainingSteps = size(X_train,2);
    validationSteps = size(X_val,2);
    E_trs = [];
    E_vals = [];
    
    fprintf('Input scaling: %.2f - Reservoir dimension: %d Spectral radius: %.2f - Lambda: %.4f\n', omega_in, Nr, rho, l);
%     fprintf('Input scaling: %.2f - Reservoir dimension: %d Spectral radius: %.2f - Lambda: %.4f - Connectivity percentage: %.2f\n', omega_in, Nr, rho, l, c);
    
    for n = 1:guesses        
        % initialize the input-to-reservoir matrix
        U = 2*rand(Nr,Nu+1)-1;
        U = omega_in * U;
        % initialize the inter-reservoir weight matrices
        W = 2*rand(Nr,Nr) - 1;
        W = rho * (W / max(abs(eig(W))));
        state = zeros(Nr,1);
        H = [];
        
        % run the reservoir on the input stream
        for t = 1:trainingSteps
            state = tanh(U * [X_train(t);1] + W * state);
            H(:,end+1) = state;
        end
        % discard the washout
        H = H(:,Nr+1:end);
        % add the bias
        H = [H;ones(1,size(H,2))];
        % update the target matrix dimension
        D = Y_train(:,Nr+1:end);
        % train the readout
        V = D*H'*inv(H*H'+ l * eye(Nr+1));
        
        % compute the output and error (loss) for the training samples
        Y_train_pred = V * H;
        err_tr = immse(D,Y_train_pred);
        E_trs(end+1) = err_tr;
        
        state = zeros(Nr,1);
        H_val = [];
        % run the reservoir on the validation stream
        for t = 1:validationSteps
            state = tanh(U * [X_val(t);1] + W * state);
            H_val(:,end+1) = state;
        end
        % add the bias
        H_val = [H_val;ones(1,size(H_val,2))];
        % compute the output and error (loss) for the validation samples
        Y_val_pred = V * H_val;
        err_val = immse(Y_val,Y_val_pred);
        E_vals(end+1) = err_val;
       
    end
    error_tr = mean(E_trs);
    Ers_tr(end+1) = error_tr;
    fprintf('Error on training set: %.5f\n', error_tr);
    error_val = mean(E_vals);
    Ers_val(end+1) = error_val;
    fprintf('Error on validation set: %.5f\n\n', error_val);
    fprintf('\n#%d/%d: ', g, size(grid,1));
end

[value, idx] = min(Ers_val);
omega_in = grid(g,1);
Nr = grid(g,2);
rho = grid(g,3);
l = grid(g,4);
fprintf('\nBest hyper-params:\nInput scaling: %.2f - Reservoir dimension: %d - Spectral radius: %.2f - Lambda: %.4f\nTraining MSE: %.5f\nValidation MSE: %.5f\n', grid(idx,1), grid(idx,2), grid(idx,3), grid(idx,4), Ers_tr(idx), value);
% fprintf('\nBest hyper-params:\nInput scaling: %.2f - Reservoir dimension: %d Spectral radius: %.2f - Lambda: %.4f - Connectivity percentage: %.2f\nValidation error: %.5f\n', grid(idx,1), grid(idx,2), grid(idx,3), grid(idx,4), grid(idx,5), value);

% model assessment
Nu = size(X_design,1);
designSteps = size(X_design,2);
testSteps = size(X_test,2);
% initialize the input-to-reservoir matrix
U = 2*rand(Nr,Nu+1)-1;
U = omega_in * U;
% initialize the inter-reservoir weight matrices
W = 2*rand(Nr,Nr) - 1;
W = rho * (W / max(abs(eig(W))));
state = zeros(Nr,1);
H = [];

% run the reservoir on the input stream
for t = 1:designSteps
    state = tanh(U * [X_design(t);1] + W * state);
    H(:,end+1) = state;
end
% discard the washout
H = H(:,Nr+1:end);
% add the bias
H = [H;ones(1,size(H,2))];
% update the target matrix dimension
D = Y_design(:,Nr+1:end);
% train the readout
V = D*H'*inv(H*H'+ l * eye(Nr+1));
% compute the output
Y_tr_pred = V * H;

state = zeros(Nr,1);
H_test = [];
% run the reservoir on the test stream
for t = 1:testSteps
    state = tanh(U * [X_test(t);1] + W * state);
    H_test(:,end+1) = state;
end
% add the bias
H_test = [H_test;ones(1,size(H_test,2))];
% compute the output and error (loss) for the validation samples
Y_test_pred = V * H_test;
error_test = immse(Y_test,Y_test_pred);
fprintf('Test MSE: %.5f\n', error_test);


time = 1:steps;
fig = figure;
tiledlayout(2,1)
% Top plot
nexttile
hold on;
plot(time,Y_design);  % ground truth
plot(1:size(Y_tr_pred,2),Y_tr_pred); % predictions
hold off;
legend('target','predictions');
title('TR+VAL target and output signals');

% Bottom plot
nexttile
hold on;
plot(time,Y_test);  % ground truth
plot(time,Y_test_pred); % predictions
hold off;
legend('target','predictions');
title('TEST target and output signals');
print(fig,'esn_target_predictions.png','-dpng')