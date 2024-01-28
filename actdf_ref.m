function res = actdf_ref(sfvalue)
% derivative of sigmoid activation function
% sfvalue - value of sigmoid activation function(!!!)

	%res = zeros(size(sfvalue));
	res = sfvalue .* (1 - sfvalue);
end
