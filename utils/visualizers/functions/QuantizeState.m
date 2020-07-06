% Quantize a denormalized state to the quantized space
function quantized = QuantizeState(x, eta, lower)
    quantized = zeros(length(x), 1);
    
    for i = 1:length(x)
       % quantized(i) = lower(i) + floor((x(i) - lower(i)) / eta(i)) * eta(i) + 0.5 * eta(i);
       quantized(i) = lower(i) + round((x(i) - lower(i)) / eta(i)) * eta(i);
    end
end