%%%%%%%%%%%%%%%%%%%% (O) Threshold Variability %%%%%%%%%%%%%%%%%%%%
% A preceding excitatory pulse might raise the threshold of a firing neuron
% and make it less excitable.

clear variables;

a=0.03;  b=0.25;  c=-60;  d=4;
j=0.04;  k=5;  l=140;
r=false;

u=-64;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.25;
tspan = 0:tau:100;

for t=tspan
    if ((t>10) && (t < 15)) || ((t>80) && (t < 85))
        I=1;
    elseif (t>70) && (t < 75)
        I=-6;
    else
        I=0;
    end
    
    [u, w, du, dw, ud, wd] = izhikevich(a, b, c, d, j, k, l, u, w, I, tau, r);
    udot(end+1)=ud;
    wdot(end+1)=wd;
    grad_u(end+1)=du;
    grad_w(end+1)=dw;
end

% plot membrane potential
fig = figure;
plot(tspan,udot,[0 10 10 15 15 70 70 75 75 80 80 85 85 max(tspan)],...
          -85+[0 0  5  5  0  0  -5 -5 0  0  5  5  0  0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(O) threshold variability');
print(fig,'img/O_threshold_variabilty_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(O) threshold variability phase portrait');
print(fig,'img/O_threshold_variabilty_phase_portrait.png','-dpng')