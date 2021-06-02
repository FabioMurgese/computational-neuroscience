%%%%%%%%%%%%%%% Learning using Covariance Rule %%%%%%%%%%%%%%%
clear variables;

data = readtable('../lab2_1_data.csv');  % importing data as table
U = table2array(data);  % converting table into input array
U_size = size(U,2);  % training set dimension
eta = 5*10e-4;  % learning rate
epochs=1000;  % iterations
Q = U*U';  % input correlation matrix
weights = [];

w = -1 + 2.*rand(2,1);  % random weights initialization
W_norm = [];

for i = 1:epochs
    U = U(:,randperm(U_size));  % reshuffling dataset
    w_temp = w;
    
    for n = 1:U_size
        % linear firing model
        u = U(:,n);
        theta = mean(u);  % update threshold for early stopping
        v = w' * u;  % compute output
        delta_w = v  * (u - theta);
        w = w + eta * delta_w;  % update weights
    end

    weights = [weights; w];
    W_norm = [W_norm; norm(w)];
    diff = norm(w - w_temp);
    
    fprintf('Epoch: %d Norm(W): %1.5f Diff: %1.7f Theta: %1.7f \n', i, norm(w), diff, theta)
    
    if diff < theta  % stop condition
        break;
    end    
end

[eigvecs, D] = eig(Q);  % computing eigenvalues and diagonal matrix of Q
eigvals = diag(D);  % storing eigenvalues in a separated array
[max_eigval, max_i] = max(eigvals);  % take the principal eigenvector index

% Plotting data points and comparison between final weight vector and
% principal eigenvector of Q
fig = figure;
hold on
plot(U(1,:),U(2,:), '.')
plotv(eigvecs(:,max_i));
set(findall(gca,'Type', 'Line'),'LineWidth',1.75);
plotv(w/norm(w))
legend('data points','principal eigenvector','weight vector','Location', 'best')
title('P1: data points, final weight vector and principal eigenvector of Q');
print(fig,'P1.png','-dpng')

w1 = weights(1:2:end);
w2 = weights(2:2:end);

% weight over time, first component
fig = figure;
plot(w1)
xlabel('time')
ylabel('weight')
title('Weight vector over time (1st component)')
print(fig,'P2.1.png','-dpng')

% weight over time, second component
fig = figure;
plot(w2)
xlabel('time')
ylabel('weight')
title('Weight vector over time (2nd component)')
print(fig,'P2.2.png','-dpng')

% weight norm over time
fig = figure;
plot(1:size(W_norm,1), W_norm)
xlabel('time')
ylabel('weight')
title('Weight norm vector over time')
print(fig,'P2.3.png','-dpng')

save('weights.mat','weights');