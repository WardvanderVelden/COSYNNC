% Perform runge kutta fourth order integration
function output = RungeKutta(ODE, x, t, u, h, steps) 
    ch = h / steps;

    for i = 1:steps
        k1 = ODE(x, t, u) * ch;
        k2 = ODE(x + k1/2, t + ch/2, u) * ch;
        k3 = ODE(x + k2/2, t + ch/2, u) * ch;
        k4 = ODE(x + k3, t + ch, u) * ch;
        
        x = x + (k1 + 2*k2 + 2*k3 + k4) * (1/6);
        t = t + ch;
    end
    
    output = x;
end