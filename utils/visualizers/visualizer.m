% A MATLAB visualizer to visualize the winning set and behaviour of the neural network controller

clc;
close all;
clear all;

addpath("functions");

%% Load neural network
run controllers/net
run controllers/dom

%% Controller settings
controlSpecification = "reachability";
isLabelled = 1;

maxIterations = 100;

h = 0.1;
addpath("plants/rocket");

xi = [-2.5 -2.5]';

useAlpha = 0;

%% Format neural network

% Format the neural network to allow for easy access
w{1} = w0; w{2} = w1; w{3} = w2; %w{4} = w3; %w{5} = w4;
b{1} = b0; b{2} = b1; b{3} = b2; %b{4} = b3; %b{5} = b4;

clear w0 w1 w2 b0 b1 b2 input; %clear w3 b3; %clear w4 b4;

% Calculate labels per axis from the quantization parameters
[stateSpaceDimension, stateSpaceLabelsPerAxis, stateSpaceIndicesPerDim] = ProcessSpace(stateSpaceEta, stateSpaceLowerBound, stateSpaceUpperBound);
[inputSpaceDimension, inputSpaceLabelsPerAxis, inputSpaceIndicesPerDim] = ProcessSpace(inputSpaceEta, inputSpaceLowerBound, inputSpaceUpperBound);


%% Find domain
domainSize = length(domain);
empiricalDomain = zeros(domainSize, 1);

for j = 1:domainSize
    goalReached = false;
    inBound = true;

    i = 1;
    
    % Get state based on index
    x = GetLowerVertexFromIndex(j - 1, stateSpaceEta, stateSpaceLowerBound, stateSpaceIndicesPerDim);
    x = x' + stateSpaceEta' * 0.5;
    
    % Check if goal is reached or out of bounds
    if(x(1) > goalLowerVertex(1) && x(1) < goalUpperVertex(1) && x(2) > goalLowerVertex(2) && x(2) < goalUpperVertex(2)) 
        if(controlSpecification == "reachability")
            goalReached = true;
            empiricalDomain(j) = 1;
        end
    else
       if(controlSpecification == "invariance")
          inBound = false;
          empiricalDomain(j) = 0;
       end
    end
    
    while (~goalReached && inBound)
        % Quantize and normalize state
        qx = QuantizeState(x, stateSpaceEta, stateSpaceLowerBound);
        nx = NormalizeState(qx, stateSpaceLowerBound, stateSpaceUpperBound);

        % Evaluate network and get input
        n = EvaluateNetwork(nx, w, b, activationFunction);
        if(isLabelled == 1)
            u = GetControlActionFromNetworkForLabelledNeurons(n, inputSpaceEta, inputSpaceLowerBound, inputSpaceIndicesPerDim);
        else
            u = GetControlActionFromNetworkForRangeNeurons(n, inputSpaceEta, inputSpaceLowerBound, inputSpaceUpperBound);
        end

        % Iterate the plant
        x = RungeKutta(@ODE, x, 0.0, u, h, 4);

        % Check if goal is reached
        if(x(1) > goalLowerVertex(1) && x(1) < goalUpperVertex(1) && x(2) > goalLowerVertex(2) && x(2) < goalUpperVertex(2)) 
            if(controlSpecification == "reachability")
                goalReached = true;
                empiricalDomain(j) = 1;
            end
        else
           if(controlSpecification == "invariance")
              inBound = false;
              empiricalDomain(j) = 0;
           end
        end
        
        % Check if out of bounds
        if(x(1) < stateSpaceLowerBound(1) || x(1) > stateSpaceUpperBound(1) || x(2) < stateSpaceLowerBound(2) || x(2) > stateSpaceUpperBound(2))
            inBound = false; 
            empiricalDomain(j) = 0;
        end

        % Check if max iterations has been reached
        if(i > maxIterations) 
            break;
        end

        i = i + 1;
    end
    
    if(controlSpecification == "invariance" || controlSpecification == "reachandstay") 
        if(x(1) > goalLowerVertex(1) && x(1) < goalUpperVertex(1) && x(2) > goalLowerVertex(2) && x(2) < goalUpperVertex(2))
            empiricalDomain(j) = 1; 
        end
    end
    
    if(~inBound) 
       empiricalDomain(j) = 0; 
    end
end

winningPercentage = sum(empiricalDomain) / domainSize * 100;
disp("Winning domain percentage: " + winningPercentage + "%");



%% Do an empirical run to confirm
x = xi;

xs = zeros(stateSpaceDimension, 1);
us = zeros(inputSpaceDimension, 1);

xs(:,1) = x;

goalReached = false;
inBound = true;

i = 1;
while (~goalReached && inBound)
    % Quantize and normalize state
    qx = QuantizeState(x, stateSpaceEta, stateSpaceLowerBound);
    nx = NormalizeState(qx, stateSpaceLowerBound, stateSpaceUpperBound);

    % Evaluate network and get input
    n = EvaluateNetwork(nx, w, b, activationFunction);
    if(isLabelled == 1)
        u = GetControlActionFromNetworkForLabelledNeurons(n, inputSpaceEta, inputSpaceLowerBound, inputSpaceIndicesPerDim);
    else
        u = GetControlActionFromNetworkForRangeNeurons(n, inputSpaceEta, inputSpaceLowerBound, inputSpaceUpperBound);
    end
    us(:,i) = u;

    % Iterate the plant
    x = RungeKutta(@ODE, x, 0.0, u, h, 4);
    xs(:,i + 1) = x;

    % Check if goal is reached or out of bounds
    if(x(1) > goalLowerVertex(1) && x(1) < goalUpperVertex(1) && x(2) > goalLowerVertex(2) && x(2) < goalUpperVertex(2) && controlSpecification == "reachability") 
        goalReached = true;
    end

    if(x(1) < stateSpaceLowerBound(1) || x(1) > stateSpaceUpperBound(1) || x(2) < stateSpaceLowerBound(2) || x(2) > stateSpaceUpperBound(2))
        inBound = false; 
    end

    if(i > maxIterations) 
        break;
    end
    
    i = i + 1;
end 



%% Plot data
faceAlpha = 0.6;
edgeAlpha = 0.6;

if(useAlpha == 0)
   faceAlpha = 1.0;
   edgeAlpha = 1.0;
end

figure('Renderer', 'painters', 'Position', [10 10 1000 600]);
hold on;

% Plot the winning domain
for i = 1:length(domain)
   index = i - 1;
   vertex = GetLowerVertexFromIndex(index, stateSpaceEta, stateSpaceLowerBound, stateSpaceIndicesPerDim);
   
   if(domain(i) == 1 && empiricalDomain(i) == 0)
        rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.4 0.0 0.9 faceAlpha], 'EdgeColor', [0.3 0.0 0.7 edgeAlpha], 'LineWidth', 0.1);
   elseif(domain(i) == 1 && empiricalDomain(i) == 1)
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.0 0.0 0.9 faceAlpha], 'EdgeColor', [0.0 0.0 0.7 edgeAlpha], 'LineWidth', 0.1);
   elseif(empiricalDomain(i) == 1)
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.0 0.9 0.2 faceAlpha], 'EdgeColor', [0.0 0.7 0.15 edgeAlpha], 'LineWidth', 0.1);
   else
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.9 0.0 0.0 faceAlpha], 'EdgeColor', [0.7 0.0 0.0 edgeAlpha], 'LineWidth', 0.1);
   end
end

% Plot goal
rectangle('Position', [goalLowerVertex(1) goalLowerVertex(2) (goalUpperVertex(1) - goalLowerVertex(1)) (goalUpperVertex(2) - goalLowerVertex(2))], 'EdgeColor', 'g', 'LineWidth', 0.5);

% Plot bounds
rectangle('Position', [stateSpaceLowerBound(1) stateSpaceLowerBound(2) (stateSpaceUpperBound(1)-stateSpaceLowerBound(1)) (stateSpaceUpperBound(2)-stateSpaceLowerBound(2))], 'EdgeColor', 'r', 'LineWidth', 0.5);

% Plot evolution
plot(xs(1,:), xs(2,:), '-wx');

% Plot settings
hold off;
axis([stateSpaceLowerBound(1)-0.1 stateSpaceUpperBound(1)+0.1 stateSpaceLowerBound(2)-0.1 stateSpaceUpperBound(2)+0.1]);
axis equal;
grid on;
title("The winning domain of the controller");
xlabel("x0");
ylabel("x1");

