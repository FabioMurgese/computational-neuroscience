%%%%%%%%%%%%%%%%%%%% (K) Resonator %%%%%%%%%%%%%%%%%%%%
% Neurons that respond only to the doulet whose frequencies resonate with
% the frequencies of subthreshold oscillations.

clear variables;

a=0.1;  b=0.26;  c=-60;  d=-1;
j=0.04;  k=5;  l=140;
r=false;

u=-62;  % threshold value of the model neuron
w=b*u;

udot=[]; 
wdot=[];
grad_u=[]; 
grad_w=[];

tau = 0.25;
tspan = 0:tau:400;
T1=tspan(end)/10;
T2=T1+20;
T3 = 0.7*tspan(end);
T4 = T3+40;

for t=tspan
    if ((t>T1) && (t < T1+4)) || ((t>T2) && (t < T2+4)) || ((t>T3) && ...
            (t < T3+4)) || ((t>T4) && (t < T4+4)) 
        I=0.65;
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
plot(tspan,udot,[0 T1 T1 (T1+8) (T1+8) T2 T2 (T2+8) (T2+8) T3 T3 (T3+8)... 
    (T3+8) T4 T4 (T4+8) (T4+8) max(tspan)],...
    -90+[0 0 10 10 0 0 10 10 0 0 10 10 0 0 10 10 0 0]);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(K) resonator');
print(fig,'img/K_resonator_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(K) resonator phase portrait');
print(fig,'img/K_resonator_phase_portrait.png','-dpng')