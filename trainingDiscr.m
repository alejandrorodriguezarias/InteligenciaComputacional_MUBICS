
function [Recall,Spec,Precision,NPV,ACC,F1Score] = trainingDiscr(typeDiscr, cv, inputs, outputs)
%% entrenamos con el discriminante lineal
for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    teIdx = cv.test(i);
    mdlListDiscrLinear{i} = fitcdiscr(inputs(trIdx,:), outputs(trIdx,:), 'DiscrimType', typeDiscr);
    prediction = predict(mdlListDiscrLinear{i}, inputs(teIdx,:));
    [CM, orderCM] = confusionmat(outputs(teIdx,:), prediction);
    for j = 1:size(CM,1)
        [Recall(i,j),Spec(i,j),Precision(i,j),NPV(i,j),ACC(i,j),F1Score(i,j)] = performance_indexes(CM,j);
    end
end