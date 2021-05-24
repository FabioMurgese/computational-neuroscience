%%%%%%%%%%%%%%% Learning using BCM Rule %%%%%%%%%%%%%%%
clear variables;

data = readtable('../lab2_1_data.csv');  % importing data as table
U = table2array(data);  % converting table into input array
U_size = size(U,2);  % training set dimension
eta = 1*10e-2;  % learning rate
epochs=1000;  % iterations
%theta = 2*10e-5;  % threshold for early stopping
%theta = NaN;
Q = U*U';  % input correlation matrix
weights = [];

w = -1 + 2.*rand(2,1);  % random weights initialization
W_norm = [];
vs = [];

for i = 1:epochs
    U = U(:,randperm(U_size));  % reshuffling dataset
    w_norm = norm(w);
    
    for n = 1:U_size
        % linear firing model
        u = U(:,n);
        v = w' * u;  % compute output
        vs = [vs; v];  % save all v to compute theta
        theta = mean(vs);
        delta_w = v  * u * (v - theta);
        w = w + eta * delta_w;  % update weights
    end
    
    theta = v^2 - theta;  % update theta
    weights(:,i) = w/norm(w);
    
    w_norm_new = norm(w);
    W_norm = [W_norm; w_norm_new];
    diff = w_norm_new - w_norm;
    
    fprintf('Epoch: %d Norm(W): %1.5f Diff: %1.7f Theta: %1.7f \n', i, w_norm, diff, theta)
    
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


x=(1:1:length(weights));  % epochs time array
% weight over time, first component
fig = figure;
plot(x, weights(1,:));
xlabel('time')
ylabel('weight')
title('Weight vector over time (1st component)')
print(fig,'P2.1.png','-dpng')

% weight over time, second component
fig = figure;
plot(x, weights(2,:))
xlabel('time')
ylabel('weight')
title('Weight vector over time (2nd component)')
print(fig,'P2.2.png','-dpng')

% weight norm over time
fig = figure;
plot(x, W_norm)
xlabel('time')
ylabel('weight')
title('Weight norm vector over time')
print(fig,'P2.3.png','-dpng')

save('weights.mat','weights');