function dataSize = CalculateDataSize(w, b)
    depth = length(w);
    floats = 0;
    
    for i = 1:depth
        [width, height] = size(w{i});
        floats = floats + width * height;
        
        height = length(b{i});
        floats = floats + height;
    end
    
    dataSize = floats*4;
    dataSize = dataSize + (2 + depth);
end
