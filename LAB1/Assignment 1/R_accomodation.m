%%%%%%%%%%%%%%%%%%%% (R) Accomodation %%%%%%%%%%%%%%%%%%%%
% Neurons are extremely sensitive to brief coincident inputs, but may not
% fire in response to a strong but slowly increasing input.

clear variables;

a=0.02;  b=1;  c=-55;  d=4;
j=0.04;  k=5;  l=140;
r=true;

u=-65;  % threshold value of the model neuron
w=-16;

udot=[]; 
wdot=[];
Idot=[];
grad_u=[]; 
grad_w=[];

tau = 0.5;
tspan = 0:tau:400;

for t=tspan
    if (t < 200)
        I=t/25;
    elseif t < 300
        I=0;
    elseif t < 312.5
        I=(t-300)/12.5*4;
    else
        I=0;
    end
    
    [u, w, du, dw, ud, wd] = izhikevich(a, b, c, d, j, k, l, u, w, I, tau, r);
    udot(end+1)=ud;
    wdot(end+1)=wd;
    grad_u(end+1)=du;
    grad_w(end+1)=dw;
   
    Idot(end+1)=I;
end

% plot membrane potential
fig = figure;
plot(tspan,udot,tspan,Idot*1.5-90);
axis([0 max(tspan) -90 30])
xlabel('time')
ylabel('membrane potential')
title('(R) accomodation');
print(fig,'img/R_accomodation_membrane_potential.png','-dpng')

% plot phase portrait
fig = figure;
hold on;
plot(udot,wdot)
quiver(udot,wdot,grad_u,grad_w,'r')
xlabel('membrane potential')
ylabel('recovery variable')
title('(R) accomodation phase portrait');
print(fig,'img/R_accomodation_phase_portrait.png','-dpng')