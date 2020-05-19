function mdlListDiscrLinear = trainingTree(cv, inputs, outputs, MaxNumSplitsValue, MinLeafSizeValue, MinParentSizeValue, MergeLeavesValue, InputsNames, SplitCriterionValue)
%% entrenamos con arboles
for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    mdlListDiscrLinear{i} = fitctree(inputs(trIdx,:), outputs(trIdx,:),'MaxNumSplits', MaxNumSplitsValue, 'MinLeafSize', MinLeafSizeValue, 'MinParentSize', MinParentSizeValue, 'MergeLeaves', MergeLeavesValue,'PredictorNames',InputsNames, 'SplitCriterion', SplitCriterionValue);
end
end

% , 'PredictorNames', PredictorNamesValue, 'ResponseName', ResponseNameValue, 'SplitCriterion', SplitCriterionValue