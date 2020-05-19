function mdlListDiscrLinear = trainingSVM(cv, inputs, outputs,BoxConstraintValue,KernelFunctionValue,ParameterValue)
%% entrenamos con la svm
for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    switch KernelFunctionValue
        case 'gaussian'
            mdlListDiscrLinear{i} = fitcsvm(inputs(trIdx,:), outputs(trIdx,:),'BoxConstraint',BoxConstraintValue, 'KernelFunction', KernelFunctionValue, 'KernelScale', ParameterValue);
        case 'polynomial'
            mdlListDiscrLinear{i} = fitcsvm(inputs(trIdx,:), outputs(trIdx,:),'BoxConstraint',BoxConstraintValue, 'KernelFunction', KernelFunctionValue, 'PolynomialOrder', ParameterValue);
        case 'linear'
            mdlListDiscrLinear{i} = fitcsvm(inputs(trIdx,:), outputs(trIdx,:),'BoxConstraint',BoxConstraintValue, 'KernelFunction', KernelFunctionValue);
        otherwise
            warning('No se reconoce el tipo de kernel')
    end
end
