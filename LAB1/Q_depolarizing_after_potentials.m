%%%%%%%%%%%%%%%%%%%% (Q) Depolarizing After-Potentials %%%%%%%%%%%%%%%%%%%%
% After spiking a spike, the membrane potential of a neuron may exhibit a
% prolonged after-hyperpolariation or a prolonged depolarized
% after-potential; this neuron has shortened refractory period and it
% becomes superexcitable.

clear variables;

a=1;  b=0.2;  c=-60;  d=-21;
j=0.04;  k=5;  l=140;
r=false;

u=-70;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.25;
tspan = 0:tau:50;
T1=10;

for t=tspan
    if abs(t-T1)<1 
        I=20;
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
plot(tspan,udot,[0 T1-1 T1-1 T1+1 T1+1 max(tspan)],-90+[0 0 10 10 0 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(Q) DAP');
print(fig,'img/Q_DAP_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(Q) DAP phase portrait');
print(fig,'img/Q_DAP_phase_portrait.png','-dpng')