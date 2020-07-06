function dxdt = ODE(x, t, u)
    dxdt = [0.5962 0.4243; 6.8197 0.7145]*x + [5.2165; 0.9673]*u;
end
