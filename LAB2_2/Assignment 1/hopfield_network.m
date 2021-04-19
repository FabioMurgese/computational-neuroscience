%%%%%%%%%%%%%%% Hopfield Neural Network %%%%%%%%%%%%%%%
clear variables;

M = load('lab2_2_data.mat');  % import data
npatterns = length(fieldnames(M));
p0 = extractfield(M, 'p0');
p1 = extractfield(M, 'p1');
p2 = extractfield(M, 'p2');

% generating distorted imges
% distorting p0
d_0_1 = distort_image(p0, 0.05);
d_0_2 = distort_image(p0, 0.1);
d_0_3 = distort_image(p0, 0.25);

% distorting p1
d_1_1 = distort_image(p1, 0.05);
d_1_2 = distort_image(p1, 0.1);
d_1_3 = distort_image(p1, 0.25);

% distorting p2
d_2_1 = distort_image(p2, 0.05);
d_2_2 = distort_image(p2, 0.1);
d_2_3 = distort_image(p2, 0.25);

% storage phase
W = [p0; p1; p2];
W = (W' * W)/size(W,2);

I = ones(length(p0),1)*0.5;  % bias
W = W - diag(diag(W));  % 0 diagonal elements

% retrieval phase
eps = 0.5;  % threshold for stopping condition
U = [d_0_1; d_0_2; d_0_3; d_1_1; d_1_2; d_1_3; d_2_1; d_2_2; d_2_3];  % distorted images
O = U;  % copy vector for original distorted images
L = [[0, 0.05]; [0, 0.10]; [0, 0.25]; [1, 0.05]; [1, 0.10]; [1, 0.25]; [2, 0.05]; [2, 0.10]; [2, 0.25]];  % pattern-distortion info matrix
patterns = [p0; p0; p0; p1; p1; p1; p2; p2; p2];

for p = 1:size(L,1)
    fprintf('Pattern %d Distortion %0.2f \n', L(p,1), L(p,2));
    
    overlap_p0 = []; overlap_p1 = []; overlap_p2 = []; energies = []; energy_old = energy(W, U(p), I);
    
    % iteration until convergence
    while(true)
        % permuting the neurons for random update
        idxs = randperm(size(W,2));
        for i = idxs
            % update equation
            U(p,i) = sign(W(i,:) * U(p,:)' + I(i));
            
            % overlap functions
            overlap_p0(end+1) = overlap(p0,U(p,:));
            overlap_p1(end+1) = overlap(p1,U(p,:));
            overlap_p2(end+1) = overlap(p2,U(p,:));
            
            % energy function
            energies(end+1) = energy(W, U(p,:), I);
        end

        % evaluating stopping condition
        energy_new = energy(W, U(p,:), I);
        if abs(energy_new - energy_old) < eps
            break;
        end
        energy_old = energy_new;
    end
    
    % plots
    fig = figure;
    hold on;
    plot((1:size(overlap_p0,2)),overlap_p0);
    plot((1:size(overlap_p1,2)),overlap_p1);
    plot((1:size(overlap_p2,2)),overlap_p2);
    title(sprintf('Overlaps (pattern %d)', L(p,1)));
    xlabel('time')
    ylabel('overlap');
    legend('pattern 0','pattern 1','pattern 2');
    print(fig,sprintf('img/distorted_%d_%0.2f_overlap.png', L(p,1), L(p,2)),'-dpng');
    hold off;
    fig = figure;
    plot((1:size(energies,2)),energies);
    title(sprintf('Energy function (pattern %d)', L(p,1)));
    xlabel('time')
    ylabel('energy function');
    print(fig,sprintf('img/distorted_%d_%0.2f_energy.png', L(p,1), L(p,2)),'-dpng');
    fig = figure;
    imagesc([reshape(O(p,:),32,32) reshape(U(p,:),32,32)])
    title(sprintf('Pattern %d reconstructed (hamming distance=%d)', L(p,1), hamming_distance(U(p,:), patterns(p,:))));
    print(fig,sprintf('img/distorted_%d_%0.2f_reconstructed.png', L(p,1), L(p,2)),'-dpng');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = overlap(p, x)
    m = (p * x') / size(p, 2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e = energy(W, x, I)
    e = -(1/2) * ((x * W * x') - (x * I));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d = hamming_distance(x, y)
    d = 0;
    for i=1:length(x)
        if x(i) ~= y(i)
            d = d + 1;
        end
    end
end