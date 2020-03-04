
function mdlListDiscrLinear = trainingDiscr(typeDiscr, cv, inputs, outputs)
%% entrenamos con el discriminante
for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    mdlListDiscrLinear{i} = fitcdiscr(inputs(trIdx,:), outputs(trIdx,:), 'DiscrimType', typeDiscr);
end
end