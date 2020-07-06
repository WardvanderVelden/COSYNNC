% Find input based on network output for labelled output type
function output = GetControlActionFromNetworkForLabelledNeurons(n, eta, lower, indicesPerDim) 
    %[~, index] = max(n);
    %index = index - 1;
    
    highest = 0.0;
    highestIndex = 1;
    for i=1:length(n)
        if(n(i) > highest)
           highest = n(i);
           highestIndex = i;
        end
    end
    index = highestIndex - 1;
    
    dim = length(eta);
    output = zeros(dim, 1);
    
    if(dim == 1)
        output(1) = lower(1) + index*eta(1);
    else
        for i=dim:-1:1
            indexOnAxis = floor(index / indicesPerDim(i));
            output(i) = lower(i) + indexOnAxis * eta(i);
            
            index = index - indexOnAxis * indicesPerDim(i);
        end
    end
    
    %s = sum(abs(output));
    %output = abs(output) / s;
end