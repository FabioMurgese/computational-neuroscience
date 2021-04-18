%%%%%%%%%%%%%%% Hopfield Neural Network %%%%%%%%%%%%%%%
clear variables;

M = load('lab2_2_data.mat');  % import data
npatterns = length(fieldnames(M));
p0 = extractfield(M, 'p0');
p1 = extractfield(M, 'p1');
p2 = extractfield(M, 'p2');

% generating distorted imges
% distorting p0
d0_1 = distort_image(p0, 0.05);
d0_2 = distort_image(p0, 0.1);
d0_3 = distort_image(p0, 0.25);

% distorting p1
d1_1 = distort_image(p1, 0.05);
d1_2 = distort_image(p1, 0.1);
d1_3 = distort_image(p1, 0.25);

% distorting p2
d2_1 = distort_image(p2, 0.05);
d2_2 = distort_image(p2, 0.1);
d2_3 = distort_image(p2, 0.25);


% storage phase
W = [p0; p1; p2];
W = (W' * W)/size(W,2);

I = ones(length(p0),1)*0.5;  % bias
W = W - diag(diag(W));  % 0 diagonal elements


% retrieval phase
d_0_1 = d0_1; d_0_1_overlap_p0 = []; d_0_1_overlap_p1 = []; d_0_1_overlap_p2 = []; d_0_1_energy = []; d_0_1_energy_old = energy(W, d_0_1, I);
d_0_2 = d0_2; d_0_2_overlap_p0 = []; d_0_2_overlap_p1 = []; d_0_2_overlap_p2 = []; d_0_2_energy = []; d_0_2_energy_old = energy(W, d_0_2, I);
d_0_3 = d0_3; d_0_3_overlap_p0 = []; d_0_3_overlap_p1 = []; d_0_3_overlap_p2 = []; d_0_3_energy = []; d_0_3_energy_old = energy(W, d_0_3, I);

d_1_1 = d1_1; d_1_1_overlap_p0 = []; d_1_1_overlap_p1 = []; d_1_1_overlap_p2 = []; d_1_1_energy = []; d_1_1_energy_old = energy(W, d_1_1, I);
d_1_2 = d1_2; d_1_2_overlap_p0 = []; d_1_2_overlap_p1 = []; d_1_2_overlap_p2 = []; d_1_2_energy = []; d_1_2_energy_old = energy(W, d_1_2, I);
d_1_3 = d1_3; d_1_3_overlap_p0 = []; d_1_3_overlap_p1 = []; d_1_3_overlap_p2 = []; d_1_3_energy = []; d_1_3_energy_old = energy(W, d_1_3, I);

d_2_1 = d2_1; d_2_1_overlap_p0 = []; d_2_1_overlap_p1 = []; d_2_1_overlap_p2 = []; d_2_1_energy = []; d_2_1_energy_old = energy(W, d_2_1, I);
d_2_2 = d2_2; d_2_2_overlap_p0 = []; d_2_2_overlap_p1 = []; d_2_2_overlap_p2 = []; d_2_2_energy = []; d_2_2_energy_old = energy(W, d_2_2, I);
d_2_3 = d2_3; d_2_3_overlap_p0 = []; d_2_3_overlap_p1 = []; d_2_3_overlap_p2 = []; d_2_3_energy = []; d_2_3_energy_old = energy(W, d_2_3, I);


% iteration until convergence
eps = 0.5;  % threshold

% pattern 0 (d_0_1)
while(true)
    % permuting the neurons for random update
    idxs = randperm(size(W,2));
    for i = idxs
        d_0_1(i) = sign(W(i,:) * d_0_1' + I(i));
        d_0_1_overlap_p0(end+1) = overlap(p0,d_0_1);
        d_0_1_overlap_p1(end+1) = overlap(p1,d_0_1);
        d_0_1_overlap_p2(end+1) = overlap(p2,d_0_1);
        d_0_1_energy(end+1) = energy(W, d_0_1, I);
    end
    
    d_0_1_energy_new = energy(W, d_0_1, I);
    if abs(d_0_1_energy_new - d_0_1_energy_old) < eps
        break;
    end
    d_0_1_energy_old = d_0_1_energy_new;
end

fprintf('Generating images . . . \n');
% plots pattern 0 (d_0_1)
fig = figure;
hold on;
plot((1:size(d_0_1_overlap_p0,2)),d_0_1_overlap_p0);
plot((1:size(d_0_1_overlap_p1,2)),d_0_1_overlap_p1);
plot((1:size(d_0_1_overlap_p2,2)),d_0_1_overlap_p2);
title('Overlaps (pattern 0)');
xlabel('time')
ylabel('overlap');
legend('pattern 0','pattern 1','pattern 2');
print(fig,'img/distorted_0_1_overlap.png','-dpng');
hold off;
fig = figure;
plot((1:size(d_0_1_energy,2)),d_0_1_energy);
title('Energy function (pattern 0)');
xlabel('time')
ylabel('energy function');
print(fig,'img/distorted_0_1_energy.png','-dpng');
fig = figure;
imagesc([reshape(d0_1,32,32) reshape(d_0_1,32,32)])
title(sprintf('Pattern 0 reconstructed (hamming distance=%d)',hamming_distance(d_0_1, p0)));
print(fig,'img/distorted_0_1_reconstructed.png','-dpng');


% pattern 0 (d_0_2)
while(true)
    % permuting the neurons for random update
    idxs = randperm(size(W,2));
    for i = idxs
        d_0_2(i) = sign(W(i,:) * d_0_2' + I(i));
        d_0_2_overlap_p0(end+1) = overlap(p0,d_0_2);
        d_0_2_overlap_p1(end+1) = overlap(p1,d_0_2);
        d_0_2_overlap_p2(end+1) = overlap(p2,d_0_2);
        d_0_2_energy(end+1) = energy(W, d_0_2, I);
    end
    
    d_0_2_energy_new = energy(W, d_0_2, I);
    if abs(d_0_2_energy_new - d_0_2_energy_old) < eps
        break;
    end
    d_0_2_energy_old = d_0_2_energy_new;    
end

% plots pattern 0 (d_0_2)
fig = figure;
hold on;
plot((1:size(d_0_2_overlap_p0,2)),d_0_2_overlap_p0);
plot((1:size(d_0_2_overlap_p1,2)),d_0_2_overlap_p1);
plot((1:size(d_0_2_overlap_p2,2)),d_0_2_overlap_p2);
title('Overlaps (pattern 0)');
xlabel('time')
ylabel('overlap');
legend('pattern 0','pattern 1','pattern 2');
print(fig,'img/distorted_0_2_overlap.png','-dpng');
hold off;
fig = figure;
plot((1:size(d_0_2_energy,2)),d_0_2_energy);
title('Energy function (pattern 0)');
xlabel('time')
ylabel('energy function');
print(fig,'img/distorted_0_2_energy.png','-dpng');
fig = figure;
imagesc([reshape(d0_2,32,32) reshape(d_0_2,32,32)])
title(sprintf('Pattern 0 reconstructed (hamming distance=%d)',hamming_distance(d_0_2, p0)));
print(fig,'img/distorted_0_2_reconstructed.png','-dpng');

% pattern 0 (d_0_3)
while(true)
    % permuting the neurons for random update
    idxs = randperm(size(W,2));
    for i = idxs
        d_0_3(i) = sign(W(i,:) * d_0_3' + I(i));
        d_0_3_overlap_p0(end+1) = overlap(p0,d_0_3);
        d_0_3_overlap_p1(end+1) = overlap(p1,d_0_3);
        d_0_3_overlap_p2(end+1) = overlap(p2,d_0_3);
        d_0_3_energy(end+1) = energy(W, d_0_3, I); 
    end
    
    d_0_3_energy_new = energy(W, d_0_3, I);
    if abs(d_0_3_energy_new - d_0_3_energy_old) < eps
        break;
    end
    d_0_3_energy_old = d_0_3_energy_new;    
end

% plots pattern 0 (d_0_3)
fig = figure;
hold on;
plot((1:size(d_0_3_overlap_p0,2)),d_0_3_overlap_p0);
plot((1:size(d_0_3_overlap_p1,2)),d_0_3_overlap_p1);
plot((1:size(d_0_3_overlap_p2,2)),d_0_3_overlap_p2);
title('Overlaps (pattern 0)');
xlabel('time')
ylabel('overlap');
legend('pattern 0','pattern 1','pattern 2');
print(fig,'img/distorted_0_3_overlap.png','-dpng');
hold off;
fig = figure;
plot((1:size(d_0_3_energy,2)),d_0_3_energy);
title('Energy function (pattern 0)');
xlabel('time')
ylabel('energy function');
print(fig,'img/distorted_0_3_energy.png','-dpng');
fig = figure;
imagesc([reshape(d0_3,32,32) reshape(d_0_3,32,32)])
title(sprintf('Pattern 0 reconstructed (hamming distance=%d)',hamming_distance(d_0_3, p0)));
print(fig,'img/distorted_0_3_reconstructed.png','-dpng');

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