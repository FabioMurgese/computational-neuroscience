%%%%%%%%%%%%%%%%%%%%%%%%% (D) Phasic Bursting %%%%%%%%%%%%%%%%%%%%%%%%%
% Some neurons report the beginning of the simulation by transmitting a
% burst.

clear variables;

a=0.02; b=0.25; c=-55;  d=0.05;
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
    if (t>T1) 
        I=0.6;
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
title('(D) phasic bursting');
print(fig,'img/D_phasic_bursting_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(D) phasic bursting phase portrait');
print(fig,'img/D_phasic_bursting_phase_portrait.png','-dpng')