function res = actf_ref(tact)
% sigmoid activation function
% tact - total activation 

	%res = ones(size(tact));
	
	sigmoidFunc = @(z) 1./(1 + exp(-z));
	res = sigmoidFunc(tact);
end
