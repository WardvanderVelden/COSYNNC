function [dim, labelsPerAxis, indicesPerDim] = ProcessSpace(eta, lower, upper)
    dim = length(eta);
    
    labelsPerAxis = zeros(dim, 1);
    indicesPerDim = zeros(dim, 1);
    for i = 1:dim
        labelsPerAxis(i) = abs(ceil(upper(i) / eta(i)) - floor(lower(i) / eta(i))) + 1;
        
        if(i == 1) 
            indicesPerDim(i) = 1;
        else
            indicesPerDim(i) = labelsPerAxis(i - 1) * indicesPerDim(i - 1);
        end
    end
end

