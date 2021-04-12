%%%%%%%%%%%%%%%%%%%% (I) Spike Latency %%%%%%%%%%%%%%%%%%%%
% Regular spiking cells in mammalian cortex can have latencies of tens of
% ms; such latencies provide a spike-timing mechanism to encode the
% strength of the input.

clear variables;

a=0.02;  b=0.2;  c=-65;  d=6;
j=0.04;  k=5;  l=140;
r=false;

u=-70;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.2;
tspan = 0:tau:100;
T1=tspan(end)/10;

for t=tspan
    if t>T1 && t < T1+3 
        I=7.04;
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
plot(tspan,udot,[0 T1 T1 T1+3 T1+3 max(tspan)],-90+[0 0 10 10 0 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(I) spike latency');
print(fig,'img/I_spike_latency_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(I) spike latency phase portrait');
print(fig,'img/I_spike_latency_phase_portrait.png','-dpng')