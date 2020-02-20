%% limpiar  
clear all

%% cargamos la base de datos de Iris
run loadIris
inputs = [VarName1, VarName2, VarName3, VarName4]
outputs = Irissetosa
%% normalización

%% 10-fold como partición del conjunto de datos
typeDiscr = 'linear';
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);
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
%% muestra de resultados medios
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(Precision)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(Recall)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACC)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(Spec)))



