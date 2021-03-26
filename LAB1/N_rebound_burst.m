%%%%%%%%%%%%%%%%%%%% (N) Rebound Burst %%%%%%%%%%%%%%%%%%%%
% Some neurons may fire post-inhibitory bursts, that is believed that such
% bursts contribute to the sleep oscillations in the thalamo-cortical
% system.

clear variables;

a=0.03;  b=0.25;  c=-52;  d=0;
j=0.04;  k=5;  l=140;
r=false;

u=-64;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.2;
tspan = 0:tau:200;
T1=20;

for t=tspan
    if (t>T1) && (t < T1+5)
        I=-15;
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
plot(tspan,udot,[0 T1 T1 (T1+5) (T1+5) max(tspan)],-85+[0 0 -5 -5 0 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(N) rebound burst');
print(fig,'img/N_rebound_burst_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(N) rebound burst phase portrait');
print(fig,'img/N_rebound_burst_phase_portrait.png','-dpng')