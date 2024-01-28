function layers = crann(layersSizeVector)
% generates hidden and output ANN weight matrices
% cfeat - number of features. Placed at beggining in layersSizeVectorVector vector
% chn - number of neurons in the  hidden layer. There can be several layers. Placed in between cfeat and cclass.
% cclass - number of neurons in the outpur layer (= number of classes). Placed at the end.
% layersSizeVector - layers vector.

% ATTENTION: we assume that constant value (bias) IS NOT INCLUDED

	numLayers = numel(layersSizeVector); % last layer is the output size
	for i=1:numLayers-1
		layers{i} = (rand(layersSizeVector(i) + 1, layersSizeVector(i+1)) - 0.5) / sqrt(layersSizeVector(i) + 1);
	end
endfunction