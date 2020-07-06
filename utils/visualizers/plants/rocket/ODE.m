function dxdt = ODE(x, t, u)
    mass = 267;
    g = -9.81;
    
    dxdt = zeros(2,1);

    dxdt(1) = x(2);
    dxdt(2) = u(1) / mass + g;
end