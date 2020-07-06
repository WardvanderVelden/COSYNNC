% Normalize denormalized state to normalized state
function normal = NormalizeState(x, lower, upper)
    normal = zeros(length(x), 1);
    for i = 1:length(x)
       normal(i) = (x(i) - lower(i)) / (upper(i) - lower(i)); 
    end
end