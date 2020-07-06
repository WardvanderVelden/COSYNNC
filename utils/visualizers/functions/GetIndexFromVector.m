function index = GetIndexFromVector(x, spaceLowerBound, spaceEta, spaceIndicesPerDim)
    dim = length(x);
    
    index = 0;
    
    for i = dim:-1:1
        index = index + floor((x(i) - spaceLowerBound(i)) / spaceEta(i)) * spaceIndicesPerDim(i);
    end
end

