function ann_training_mod_norm(learningRate, noHiddenNeurons, noEpochs)

% load libraries needed to spit out some excel sheets to make things quicker with creating a report
pkg load io
pkg load windows

% read dataset, show and correct labels
[tvec tlab tstv tstl] = readSets(); 
tlab += 1;
tstl += 1;

rand();
% rstate = rand("state");
% save rnd_state.txt rstate
load rnd_state.txt 
rand("state", rstate);

% normalization

mn = mean(tvec);
sd = std(tvec);
sd(sd==0) = 1;
tvec = bsxfun(@minus,tvec,mn);
tvec = bsxfun(@rdivide,tvec,sd);
mn = mean(tstv);
sd = std(tstv);
sd(sd==0) = 1;
tstv = bsxfun(@minus,tstv,mn);
tstv = bsxfun(@rdivide,tstv,sd);

network = crann([columns(tvec) noHiddenNeurons numel(unique(tlab))]);
trainCorrectness = zeros(1, noEpochs); 
testCorrectness = zeros(1, noEpochs);
bestTestCorrectness = 0;
stagnationCounter = 0;
trReport = [];
for epoch=1:noEpochs

	if (stagnationCounter > 7)
		break;
	endif

	tic();
	for i=1:rows(tvec)
		[network terrN] = backprop_ref(tvec(i, :), tlab(i, :), network, learningRate);,
	end
	clsRes = anncls_ref(tvec, network);
	cfmx = confMx(tlab, clsRes);
	errcf = compErrors(cfmx);
	trainCorrectness(epoch) = 1-errcf(2); % easier to spot the better result

	clsRes = anncls_ref(tstv, network);
	cfmx = confMx(tstl, clsRes);
	errcf2 = compErrors(cfmx);
	testCorrectness(epoch) = 1-errcf2(2); % easier to spot the better result
	epochTime = toc();

	improvement = testCorrectness(epoch) - bestTestCorrectness;
	if (improvement > 0)	
		bestTestCorrectness = testCorrectness(epoch);
		if(testCorrectness(epoch) > 0.8742)
			disp(strcat("Significant improvement! ", " Value: ", num2str(testCorrectness(epoch))," Epoch: ", int2str(epoch), " Improvement in pp: ", num2str(improvement)));
			xlswrite(strcat("CFMX_N_E",int2str(epoch),"_R", num2str(learningRate), "_L", num2str(noHiddenNeurons(1)), "_", num2str(noHiddenNeurons(2)), ".xlsx"),cfmx,'CFMX','');
		else
			disp(strcat("Minor Improvement! ", " Value: ", num2str(testCorrectness(epoch))," Epoch: ", int2str(epoch), " Improvement in pp: ", num2str(improvement)));
		endif
		stagnationCounter = 0;
	else
		disp(strcat("No improvement! ", " Value: ", num2str(testCorrectness(epoch))," Epoch: ", int2str(epoch)));			
		stagnationCounter += 1
    endif
	%disp([epoch epochTime trainCorrectness(epoch) testCorrectness(epoch)])
	trReport = [trReport; epoch epochTime trainCorrectness(epoch) testCorrectness(epoch)];
	fflush(stdout);
endfor
xlswrite(strcat("TRAINING_REPORT_N", "_R", num2str(learningRate), "_L", num2str(noHiddenNeurons(1)), "_", num2str(noHiddenNeurons(2)), ".xlsx"),trReport,'Training Report','');
end
