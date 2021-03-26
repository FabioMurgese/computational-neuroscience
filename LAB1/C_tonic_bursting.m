%%%%%%%%%%%%%%%%%%%%%%%%% (C) Tonic Bursting %%%%%%%%%%%%%%%%%%%%%%%%%
% Some neurons (ie chattering neurons in cat neocortex) fire periodic
% bursts of spikes when stimulated.

clear variables;

a=0.02;  b=0.2;  c=-50;  d=2;
j=0.04;  k=5;  l=140;
r=false;

u=-70;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.25;
tspan = 0:tau:220;
T1=22;

for t=tspan
    if (t>T1) 
        I=15;
    else
        I=0;
    end
    
    [u, w, du, dw] = izhikevich(a, b, c, d, j, k, l, u, w, I, tau, r);
    grad_u(end+1)=du;
    grad_w(end+1)=dw;
    
    if u > 30  % not a threshold, but the peak of the spike
        udot(end+1)=30;
    else
        udot(end+1)=u;
    end
    wdot(end+1)=w;
end

% plot membrane potential
fig = figure;
plot(tspan,udot,[0 T1 T1 max(tspan)],-90+[0 0 10 10]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(C) tonic bursting');
print(fig,'img/C_tonic_bursting_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(C) tonic bursting phase portrait');
print(fig,'img/C_tonic_bursting_phase_portrait.png','-dpng')