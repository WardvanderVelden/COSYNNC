function dxdt = ODE(x, t, u)
    if(u(1) < 1.0)
        dxdt = [1 0; 0 -1.5] * x;
    else
        dxdt = [-1.5 0; 0 1] * x;
    end
end
