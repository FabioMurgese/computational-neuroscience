%%%%%%%%%%%%%%%%%%%% (H) Class 2 Excitable %%%%%%%%%%%%%%%%%%%%
% Some neurons cannot fire low-frequency spike trains, so they are either
% quiescent or fire a train of spikes with a certain relatively large
% frequency.

clear variables;

a=0.2;  b=0.26;  c=-65;  d=0;
j=0.04;  k=5;  l=140;
r=false;

u=-64;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.25;
tspan = 0:tau:300;
T1=30;

for t=tspan
    if (t>T1) 
        I=-0.5+(0.015*(t-T1));
    else
        I=-0.5;
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
plot(tspan,udot,[0 T1 max(tspan) max(tspan)],-90+[0 0 20 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(H) class 2 excitable');
print(fig,'img/H_class_2_excitable_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(H) class 2 excitable phase portrait');
print(fig,'img/H_class_2_excitable_phase_portrait.png','-dpng')