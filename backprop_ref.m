function [outNetwork terr] = backprop_ref(tset, tslb, inNetwork, lr)
% derivative of sigmoid activation function
% tset - training set (every row represents act sample)
% tslb - column vector of labels 
% inNet - initial layers weight matrix
% lr - learning rate

% inNetwork - layers weight matrix
% terr - total squared error of the ANN

	numLayers = numel(inNetwork) + 1;
	numLabels = columns(inNetwork{numLayers-1});
	tsetRows = rows(tset);

	%1. Propagate input forward through the ANN
	
	act{1} = tset;
	for i=2:numLayers
		response{i} = [act{i-1} ones(tsetRows, 1)] * inNetwork{i-1};
		act{i} = actf_ref(response{i});
	end

	for i=2:numLayers
		networkGrad{i-1} = zeros(size(inNetwork{i-1}));
	end

	desiredOutput = zeros(tsetRows, numLabels);

	for i=1:tsetRows
		desiredOutput(i, tslb(i)) = 1;
	end
	d{numLayers} = desiredOutput - response{numLayers};
	for i=numLayers-1: -1: 1
		d{i} = (d{i+1} * inNetwork{i}') .* [actdf_ref(act{i}) ones(tsetRows, 1)];
		d{i} = d{i}(:, 1:end-1);
		D{i} = d{i+1}' * [act{i} ones(tsetRows, 1)];
		networkGrad{i} = lr * D{i}';
	end
	terr = 0.5*sum((act{numLayers}-desiredOutput)(:).^2)/tsetRows;

	for i=2:numLayers
		outNetwork{i-1} = inNetwork{i-1} + networkGrad{i-1};
	end
end
