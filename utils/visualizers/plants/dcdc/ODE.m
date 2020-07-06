function dxdt = ODE(x, t, u)
    % Parameters
    xc = 70;
    xl = 3;
    rc = 0.005;
    rl = 0.05;
    r0 = 1;
    vs = 1;

    % Calculate derivative
    dxdt = zeros(2, 1);
    
    if (u < 0.5)
        dxdt(1) = (-rl / xl) * x(1) + vs / xl;
        dxdt(2) = ((-1 / xc) * (1 / (r0 + rc))) * x(2);
    else
        dxdt(1) = ((-1 / xl) * (rl + ((r0 * rc) / (r0 + rc)))) * x(1) + (((-1 / xl) * (r0 / (r0 + rc))) / 5) * x(2) + vs / xl;
        dxdt(2) = (5 * (r0 / (r0 + rc)) * (1 / xc)) * x(1) + ((-1 / xc) * (1 / (r0 + rc))) * x(2);
    end
end
