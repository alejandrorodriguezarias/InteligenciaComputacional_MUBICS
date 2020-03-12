%% limpiar  
clear all
%cambiar la semilla 
rng('shuffle')
%% cargamos la base de datos de Iris
run loadIris
%aleatorizamos los valores
inputs = [VarName1, VarName2, VarName3, VarName4];
rdnIndx = randperm(size(inputs,1),size(inputs,1));
inputs = inputs(rdnIndx,:);
outputs = Irissetosa(rdnIndx,:);

%% normalizaciï¿½n
% inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
inputs = normalize(inputs);
boxplot(inputs);

%corrcoef(inputs);
% eliminamos la variable anchura del petalo por estar altamente 
% correlacionada con la longitud del petado (0,96)
%inputs(:,4) = [];
%corrcoef(inputs);
%% 10-fold como particiï¿½n del conjunto de datos
typeDiscr = 'linear';
TypeCV = 'LeaveOut';
cv = cvpartition(outputs,TypeCV);
%% entrenamos con el discriminante lineal
mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% mostramos datos de entrenamiento
[RecallLinear,SpecLinear,PrecisionLinear,NPVLinear,ACCLinear,F1ScoreLinear, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 1);
fprintf('Datos de entrenamiento\n')
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(PrecisionLinear)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(RecallLinear)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACCLinear)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(SpecLinear)))

%% muestra de resultados de test
[RecallLinear,SpecLinear,PrecisionLinear,NPVLinear,ACCLinear,F1ScoreLinear, predictionLinear] = predictResultsLOO(cv, inputs, outputs, mdls);
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(PrecisionLinear)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(RecallLinear)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACCLinear)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(SpecLinear)))


%% entrenamos con el discriminante cuadratico
typeDiscr = 'quadratic';
mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% mostramos datos de entrenamiento
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = predictResults(cv, inputs, outputs, mdls,1);
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para el discriminante cuadratico con Iris: %f\n',mean(mean(PrecisionQuadr)))
fprintf('Recall media para el discriminante cuadratico con Iris: %f\n',mean(mean(RecallQuadr)))
fprintf('ACC media para el discriminante cuadratico con Iris: %f\n',mean(mean(ACCQuadr)))
fprintf('Spec media para el discriminante cuadratico con Iris : %f\n',mean(mean(SpecQuadr)))

%% muestra de resultados medios
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = predictResultsLOO(cv, inputs, outputs, mdls);
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante cuadratico con Iris: %f\n',mean(mean(PrecisionQuadr)))
fprintf('Recall media para el discriminante cuadratico con Iris: %f\n',mean(mean(RecallQuadr)))
fprintf('ACC media para el discriminante cuadratico con Iris: %f\n',mean(mean(ACCQuadr)))
fprintf('Spec media para el discriminante cuadratico con Iris : %f\n',mean(mean(SpecQuadr)))

%% 
inputsNames = {'sepalL','sepalW','petalL','petalW'};
mdls1 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,1,10,'on',inputsNames,'gdi'); % por defecto
% mdls2 = trainingTree(cv,inputs,outputs,20,5,30,'on',inputsNames,'gdi'); 
% mdls3 = trainingTree(cv,inputs,outputs,3,20,300,'on',inputsNames,'twoing');

mdlMatrix = [mdls1];

for n = 1:size(mdlMatrix,1)
    [RecallTreeTMP,SpecTreeTMP,PrecisionTreeTMP,NPVTreeTMP,ACCTreeTMP,F1ScoreTreeTMP, predictionTree] = predictResultsLOO(cv, inputs, outputs, mdlMatrix(n,:));
    RecallTree(n,:) = mean(RecallTreeTMP,2);
    SpecTree(n,:) = mean(SpecTreeTMP,2);
    PrecisionTree(n,:) = mean(PrecisionTreeTMP,2);
    ACCTree(n,:) = mean(ACCTreeTMP,2);
    F1ScoreTree(n,:) = mean(F1ScoreTreeTMP,2);
end

%% Muestra los resultados medios en entrenamiento
fprintf('\nENTRENAMIENTO');

for v = 1:size(mdlMatrix,1)
    [RecallTreeTMP,SpecTreeTMP,PrecisionTreeTMP,NPVTreeTMP,ACCTreeTMP,F1ScoreTreeTMP, predictionTree] = predictResults(cv, inputs, outputs, mdlMatrix(1,:), 1);
    fprintf('\nDatos de entrenamiento de árbol %d\n',v)
    fprintf('Precision media para árbol %d con Cancer: %f\n',v,mean(PrecisionTreeTMP))
    fprintf('Recall media para árbol %d con Cancer: %f\n',v,mean(RecallTreeTMP))
    fprintf('ACC media para árbol %d con Cancer: %f\n',v,mean(ACCTreeTMP))
    fprintf('Spec media para árbol %d con Cancer: %f\n',v,mean(SpecTreeTMP))
end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    fprintf('\nDatos de test de árbol %d\n',i)
    fprintf('Precision media para árbol %d con Cancer: %f\n',i,mean(PrecisionTree(i,:)))
    fprintf('Recall media para árbol %d con Cancer: %f\n',i,mean(RecallTree(i,:)))
    fprintf('ACC media para árbol %d con Cancer: %f\n',i,mean(ACCTree(i,:)))
    fprintf('Spec media para árbol %d con Cancer: %f\n',i,mean(SpecTree(i,:)))
end

view(mdls1{1},'Mode','graph')
view(mdls2{1},'Mode','graph')
view(mdls3{1},'Mode','graph')



%% Diferencias significativas entre modelos
ACCMeanLinear = mean(ACCLinear,2); % para cada modelo calculamos la ACC media para las tres clases de flor
ACCMeanQuadr = mean(ACCQuadr,2);

muestras = [ACCMeanLinear,ACCMeanQuadr];
etiquetas = ['linear';'quadra'];
[P] = testEstadistico(muestras,etiquetas,0.05);