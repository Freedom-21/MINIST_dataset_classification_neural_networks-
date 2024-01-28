function lab = anncls_ref(tset, layersWeights)
% simple ANN classifier
% tset - data to be classified (every row represents a sample) 
% layersWeights - layer weight matrices. Consists of several hidlw and one outlw
% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix


% lab - classification result (index of output layer neuron with highest value)
% ATTENTION: we assume that constant value IS NOT INCLUDED in tset rows

	numLayers = numel(layersWeights) + 1;
	res{1} = tset;
	for i=2:numLayers
		response{i} = [res{i-1} ones(rows(tset), 1)] * layersWeights{i-1};
		res{i} = actf_ref(response{i});
	endfor

	[~, lab] = max(res{numLayers}, [], 2);
end