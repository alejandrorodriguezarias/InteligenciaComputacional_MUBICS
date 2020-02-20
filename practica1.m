%% limpiar  
clear all

%% cargamos la base de datos de Iris
run loadIris
inputs = [VarName1, VarName2, VarName3, VarName4];
outputs = Irissetosa;
%% normalizaci�n

%% 10-fold como partici�n del conjunto de datos
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
        [RecallLinear(i,j),SpecLinear(i,j),PrecisionLinear(i,j),NPVLinear(i,j),ACCLinear(i,j),F1ScoreLinear(i,j)] = performance_indexes(CM,j);
    end
end
%% muestra de resultados medios
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(PrecisionLinear)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(RecallLinear)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACCLinear)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(SpecLinear)))


%% entrenamos con el discriminante cuadratico
typeDiscr = 'quadratic';

for i = 1:cv.NumTestSets
    trIdx = cv.training(i);
    teIdx = cv.test(i);
    mdlListDiscrQuadr{i} = fitcdiscr(inputs(trIdx,:), outputs(trIdx,:), 'DiscrimType', typeDiscr);
    prediction = predict(mdlListDiscrQuadr{i}, inputs(teIdx,:));
    [CM, orderCM] = confusionmat(outputs(teIdx,:), prediction);
    for j = 1:size(CM,1)
        [RecallQuadr(i,j),SpecQuadr(i,j),PrecisionQuadr(i,j),NPVQuadr(i,j),ACCQuadr(i,j),F1ScoreQuadr(i,j)] = performance_indexes(CM,j);
    end
end
%% muestra de resultados medios
fprintf('Precision media para el discriminante cuadratico con Iris: %f\n',mean(mean(PrecisionQuadr)))
fprintf('Recall media para el discriminante cuadratico con Iris: %f\n',mean(mean(RecallQuadr)))
fprintf('ACC media para el discriminante cuadratico con Iris: %f\n',mean(mean(ACCQuadr)))
fprintf('Spec media para el discriminante cuadratico con Iris : %f\n',mean(mean(SpecQuadr)))


%% Curvas ROC

%% Errores de entrenamiento y test

%% Diagramas de cajas y bigotes

%% Diferencias significativas entre modelos
ACCMeanLinear = mean(ACCLinear,2) % para cada modelo calculamos la ACC media para las tres clases de flor
ACCMeanQuadr = mean(ACCQuadr,2)
muestras = [ACCMeanLinear,ACCMeanQuadr]
% [P] = testEstadistico(prueba,['Linear  ';'Quadratico'],0.05)





