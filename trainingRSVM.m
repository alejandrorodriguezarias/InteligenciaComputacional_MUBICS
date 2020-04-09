function mdlListDiscrLinear = trainingRSVM(cvTraining, inputs, outputs,BoxConstraintValue,KernelFunctionValue,ParameterValue)
%% entrenamos con la svm
for i = 1:size(cvTraining,1)
    trIdx = cvTraining(i,:);
    switch KernelFunctionValue
        case 'gaussian'
            mdlListDiscrLinear{i} = fitrsvm(inputs(trIdx,:), outputs(trIdx,:),'BoxConstraint',BoxConstraintValue, 'KernelFunction', KernelFunctionValue, 'KernelScale', ParameterValue);
        case 'polynomial'
            mdlListDiscrLinear{i} = fitrsvm(inputs(trIdx,:), outputs(trIdx,:),'BoxConstraint',BoxConstraintValue, 'KernelFunction', KernelFunctionValue, 'PolynomialOrder', ParameterValue);
        case 'linear'
            mdlListDiscrLinear{i} = fitrsvm(inputs(trIdx,:), outputs(trIdx,:),'BoxConstraint',BoxConstraintValue, 'KernelFunction', KernelFunctionValue);
        otherwise
            warning('No se reconoce el tipo de kernel')
    end
end