function dxdt = ODE(x, t, u)
    v = 1;
    omega = 2;
    
    dxdt = zeros(3,1);

    dxdt(1) = v * cos(x(3));
	dxdt(2) = v * sin(x(3));
	dxdt(3) = u(1) * omega;
end
