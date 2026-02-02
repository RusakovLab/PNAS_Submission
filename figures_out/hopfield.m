% Hopfield Network in MATLAB

% Define the input patterns
patterns = [1 1 -1 -1; -1 -1 1 1; 1 -1 1 -1];

% Define the weight matrix
weights = patterns' * patterns;
diag_indices = 1:length(weights);
weights(diag_indices, diag_indices) = 0;

% Define the initial state of the network
initial_state = [1 -1 1 -1];

% Define the update function
update = @(state, weights) sign(weights .* state');

% Update the network until it converges
current_state = initial_state;
previous_state = zeros(size(current_state));
while ~isequal(current_state, previous_state)
    previous_state = current_state;
    current_state = update(current_state, weights);
end

% Display the final state of the network
disp(current_state);
