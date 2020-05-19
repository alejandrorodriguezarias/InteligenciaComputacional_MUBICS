function mdl = trainingRRNA(cvTraining, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio)
%% entrenamos con la RNA (regresion)
for i = 1:size(cvTraining,1)
    trIdx = cvTraining(i,:);
    net = patternnet(hiddenLayerSize, trainFCN, performFCN);
    net.trainParam.max_fail = 6;
    net.trainParam.min_grad = 1e-07;
    net.trainParam.epochs = 1000;
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