% A MATLAB visualizer to visualize neural network encoded winning sets

% Clear all data and add function path
clear all;
close all;
clc;

addpath("functions");


%% Load neural network encoded winning set
run winningset/net % Load neural network encoded as winning set
run controllers/dom % Load winning domain

threshold = 0.95;


%% Format neural network

% Format the neural network to allow for easy access
w{1} = w0; w{2} = w1; w{3} = w2; %w{4} = w3; %w{5} = w4;
b{1} = b0; b{2} = b1; b{3} = b2; %b{4} = b3; %b{5} = b4;

clear w0 w1 w2 b0 b1 b2 input; %clear w3 b3; %clear w4 b4;

% Calculate labels per axis from the quantization parameters
[stateSpaceDimension, stateSpaceLabelsPerAxis, stateSpaceIndicesPerDim] = ProcessSpace(stateSpaceEta, stateSpaceLowerBound, stateSpaceUpperBound);
[inputSpaceDimension, inputSpaceLabelsPerAxis, inputSpaceIndicesPerDim] = ProcessSpace(inputSpaceEta, inputSpaceLowerBound, inputSpaceUpperBound);

% Rename domain to winning set
winningSet = domain;
clear domain;

%% Find neural network winning set
neuralNetworkWinningSet = zeros(length(winningSet), 1);
fitIndices = 0;

for index = 1:length(winningSet)
    x = GetLowerVertexFromIndex(index - 1, stateSpaceEta, stateSpaceLowerBound, stateSpaceIndicesPerDim);
    x = x' + stateSpaceEta' * 0.5;
    
    % Normalize state
    nx = NormalizeState(x, stateSpaceLowerBound, stateSpaceUpperBound);

    % Evaluate neural network
    n = EvaluateNetwork(nx, w, b, activationFunction);
    
    if(n(2) >= threshold) 
        neuralNetworkWinningSet(index) = 1;
        if(winningSet(index) == 1) 
           fitIndices = fitIndices + 1; 
        end
    end
end

percentage = fitIndices / sum(winningSet) * 100;
disp("Completeness: " + percentage + "%");
%% Plot winning sets
figure('Renderer', 'painters', 'Position', [10 10 1000 600]);
hold on;

% Plot the winning domain
for index = 1:length(winningSet)
   vertex = GetLowerVertexFromIndex(index - 1, stateSpaceEta, stateSpaceLowerBound, stateSpaceIndicesPerDim);
   
   if(winningSet(index) == 1 && neuralNetworkWinningSet(index) == 1)
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.0 0.0 1.0 1.0], 'EdgeColor', [0.0 0.0 1.0 1.0], 'LineWidth', 0.1);
   elseif(winningSet(index) == 1) 
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.0 1.0 0.0 1.0], 'EdgeColor', [0.0 1.0 0.0 1.0], 'LineWidth', 0.1);    
   elseif(neuralNetworkWinningSet(index) == 1)
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [1.0 0.0 0.0 1.0], 'EdgeColor', [1.0 0.0 0.0 1.0], 'LineWidth', 0.1);
   else
       rectangle('Position', [vertex(1) vertex(2) stateSpaceEta(1) stateSpaceEta(2)], 'FaceColor', [0.0 0.0 0.0 1.0], 'EdgeColor', [0.0 0.0 0.0 1.0], 'LineWidth', 0.1);
   end
end

% Plot goal
rectangle('Position', [goalLowerVertex(1) goalLowerVertex(2) (goalUpperVertex(1) - goalLowerVertex(1)) (goalUpperVertex(2) - goalLowerVertex(2))], 'EdgeColor', 'g', 'LineWidth', 0.5);

% Plot bounds
rectangle('Position', [stateSpaceLowerBound(1) stateSpaceLowerBound(2) (stateSpaceUpperBound(1)-stateSpaceLowerBound(1)) (stateSpaceUpperBound(2)-stateSpaceLowerBound(2))], 'EdgeColor', 'r', 'LineWidth', 0.5);

% Plot settings
hold off;
axis([stateSpaceLowerBound(1)-0.1 stateSpaceUpperBound(1)+0.1 stateSpaceLowerBound(2)-0.1 stateSpaceUpperBound(2)+0.1]);
axis equal;
grid on;
title("The winning set according to the neural network");
xlabel("x0");
ylabel("x1");



