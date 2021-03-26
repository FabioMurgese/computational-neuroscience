%%%%%%%%%%%%%%%%%%%% (L) Integrator %%%%%%%%%%%%%%%%%%%%
% These neurons prefer high-frequency input; the higher the frequency the
% more likely they fire.

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
tspan = 0:tau:100;
T1=tspan(end)/11;
T2=T1+5;
T3 = 0.7*tspan(end);
T4 = T3+10;

for t=tspan
    if ((t>T1) && (t < T1+2)) || ((t>T2) && (t < T2+2)) || ((t>T3) && ...
            (t < T3+2)) || ((t>T4) && (t < T4+2))
        I=9;
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
plot(tspan,udot,[0 T1 T1 (T1+2) (T1+2) T2 T2 (T2+2) (T2+2) T3 T3 (T3+2)...
    (T3+2) T4 T4 (T4+2) (T4+2) max(tspan)],...
    -90+[0 0 10 10 0 0 10 10 0 0 10 10 0 0 10 10 0 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(L) integrator');
print(fig,'img/L_integrator_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(L) integrator phase portrait');
print(fig,'img/L_integrator_phase_portrait.png','-dpng')