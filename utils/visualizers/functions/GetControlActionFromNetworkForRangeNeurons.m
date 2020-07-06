% Find input based on network output for labelled output type
function output = GetControlActionFromNetworkForRangeNeurons(n, eta, lower, upper) 
    dim = length(eta);
    
    output = zeros(dim, 1);
    
    for i = 1:dim
        % Average neurons
        output(i) = (n((i-1)*2+1) + n((i-1)*2+2)) / 2;
       
        % Denormalize output
        output(i) = lower(i) + round(((upper(i) - lower(i)) * output(i)) / eta(i)) * eta(i); % + 0.5 * eta(i);
    end
end