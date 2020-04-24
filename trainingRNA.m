function mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio)
%% entrenamos con la RNA
for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    net = patternnet(hiddenLayerSize, trainFCN, performFCN);
    net.trainParam.max_fail = 6;
    net.trainParam.min_grad = 1e-07;
    switch trainFCN
        case 'trainlm'   
            net.trainParam.mu = 0.001;
            net.trainParam.mu_dec = 0.1;
        case 'trainscg'
            net.trainParam.sigma = 5e-05;
            net.trainParam.lambda = 5e-07;
        otherwise
            warning('No se reconoce el modelo')
    end
    % Divisiones
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = valRatio;
    net.divideParam.testRatio = 0;
    % Entrenamiento
    [mdl{i},tr] = train(net,inputs(:,trIdx),outputs(:,trIdx));
    
 
    end
end