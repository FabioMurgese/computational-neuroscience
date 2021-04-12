%%%%%%%%%%%%%%% Izhikevich model with Leap-Frog method %%%%%%%%%%%%%%%
% u: membrane potential of the neuron (activation of Na+)
% w: membrane recovery variable (inactivation of Na+)
% a: time scale of the recovery variable
% b: sensitivity of the recovery variable to fluctuations of the membrane
% potential
% c: after-spike reset value of the membrane potential
% d: after-spike rest of the recovery variable
% I: synaptic currents or injected dc-currents
% r: flag to distinguish (R) accomodation feature

function [u, w, du, dw, udot, wdot] = izhikevich(a, b, c, d, j, k, l, u, w, I, tau, r)
    
    udot=[]; 
    wdot=[];
    
    if r==true
        du = j*u^2+k*u+l-w+I;
        dw = a*(b*(u+65));
    else
        du = j*u^2+k*u+l-w+I;
        dw = a*(b*u-w);
    end
    
    u = u + tau*du;
    w = w + tau*dw;
    
    % after-spike resetting
    if u > 30  % not a threshold, but the peak of the spike
        udot(end+1)=30;
        u = c;
        w = w + d;
    else
        udot(end+1)=u;
    end
    wdot(end+1)=w;
end