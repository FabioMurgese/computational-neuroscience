%%%%%%%%%%%%%%%%%%%% (F) Spike Frequency Adaptation %%%%%%%%%%%%%%%%%%%%
% These regular spiking (RS) cells are the most common type of excitatory
% neurons in mammalian neocortex and fire tonic spikes with decreasing
% frequency.

clear variables;

a=0.01; b=0.2;  c=-65;  d=8;
j=0.04;  k=5;  l=140;
r=false;

u=-70;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.25;
tspan = 0:tau:85;
T1=tspan(end)/10;

for t=tspan
    if (t>T1) 
        I=30;
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
plot(tspan,udot,[0 T1 T1 max(tspan)],-90+[0 0 10 10]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(F) spike frequency adaptation');
print(fig,'img/F_spike_freq_adaptation_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(F) spike frequency adaptation phase portrait');
print(fig,'img/F_spike_freq_adaptation_phase_portrait.png','-dpng')