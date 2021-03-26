%%%%%%%%%%%%%%%%%%%% (G) Class 1 Excitable %%%%%%%%%%%%%%%%%%%%
% Class 1 excitable neurons can encode the strength of the input into their
% firing rate.

clear variables;

a=0.02;  b=-0.1;  c=-55;  d=6;
j=0.04;  k=4.1;  l=108;
r=false;

u=-60;  % threshold value of the model neuron
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
        I=(0.075*(t-T1));
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
plot(tspan,udot,[0 T1 max(tspan) max(tspan)],-90+[0 0 20 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(G) class 1 excitable');
print(fig,'img/G_class_1_excitable_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(G) class 1 excitable phase portrait');
print(fig,'img/G_class_1_excitable_phase_portrait.png','-dpng')