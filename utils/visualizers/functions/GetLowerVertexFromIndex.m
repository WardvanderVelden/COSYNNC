% Get a cells lower state vertex from a state index
function vertex = GetLowerVertexFromIndex(index, eta, lower, indicesPerDimension)
    dim = length(eta);

    vertex = zeros(1, length(eta));
    
    if(dim == 1)
        vertex(1) = lower(1) + index*eta(1);
    else
        for i=dim:-1:1
%             if(i ~= 1)
%                 indexOnAxis = floor(index / indicesPerDimension(i - 1));  
%                 index = index - indexOnAxis * indicesPerDimension(i - 1);
% 
%                 vertex(i) = lower(i) + indexOnAxis*eta(i) - eta(i) * 0.5;
%             else
%                 vertex(1) = lower(1) + index*eta(1) - eta(i) * 0.5;
%             end
            indexOnAxis = floor(index / indicesPerDimension(i));
            vertex(i) = lower(i) + indexOnAxis * eta(i) - eta(i) * 0.5;
            
            index = index - indexOnAxis * indicesPerDimension(i);
        end
    end
end