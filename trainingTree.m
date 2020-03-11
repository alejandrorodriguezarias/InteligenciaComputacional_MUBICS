function mdlListDiscrLinear = trainingTree(cv, inputs, outputs, MaxNumSplitsValue, MinLeafSizeValue, MinParentSizeValue, MergeLeavesValue, InputsNames)
%% entrenamos con el discriminante
for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    mdlListDiscrLinear{i} = fitctree(inputs(trIdx,:), outputs(trIdx,:),'MaxNumSplits', MaxNumSplitsValue, 'MinLeafSize', MinLeafSizeValue, 'MinParentSize', MinParentSizeValue, 'MergeLeaves', MergeLeavesValue,'PredictorNames',InputsNames);
end
end

% , PredictorNamesValue, ResponseNameValue, SplitCriterionValue
% , 'PredictorNames', PredictorNamesValue, 'ResponseName', ResponseNameValue, 'SplitCriterion', SplitCriterionValue