% Evaluates a network given a normalized input and outputs a normalized output
function output = EvaluateNetwork(x, w, b, act)
    depth = length(w);
    
    % Normalize with input factor
    % x = inputFactor' .* x;
    
    % Evaluate network
    for i = 1:depth
       if(act == 'relu')
          x = max(w{i} * x + b{i}', 0.0); 
          %x = w{i} * x + b{i}';
       end
    end
    
    % Format output
    x = min(max(x, 0.0), 1.0);
    
    % Normalize input to add to 1
    if(sum(x) > 0.001)
        output = x / sum(x);
    else
        output = x;
    end
end