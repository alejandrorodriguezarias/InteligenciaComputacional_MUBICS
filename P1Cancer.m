%% cargamos la base de datos de cancer
run loadCancer
inputs = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9];
inputsNames = {'Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'};
outputs = ClaseCancer;

%% eliminar valores nulos (se podría rellenar con valores medios)
dataset = [inputs,outputs];
%filas con valores nulos
rows = any(isnan(inputs),2);
%eliminaciï¿½n de valores nulos
dataset(rows,:) = [];
inputs = dataset(:,1:9);
outputs = dataset(:,10);
newmatrix(outputs == 2) = {'benigno'};
newmatrix(outputs == 4) = {'maligno'};
outputs = newmatrix'
% eliminamos las variables correlacionadas
corrcoef(inputs);
inputs(:,2) = [];
%% normalizaciï¿½n
% inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
inputs = normalize(inputs);
boxplot(inputs)
%% 10-fold como particion del conjunto de datos
typeDiscr = 'linear';
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);
% entrenamos con el discriminante lineal
 mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
[RecallLinear,SpecLinear,PrecisionLinear,NPVLinear,ACCLinear,F1ScoreLinear, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 1);

%% mostramos datos de entrenamiento
fprintf('\nValores para cancer\n');
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para el discriminante lineal con Cancer: %f\n',mean(PrecisionLinear))
fprintf('Recall media para el discriminante lineal con Cancer: %f\n',mean(RecallLinear))
fprintf('ACC media para el discriminante lineal con Cancer: %f\n',mean(ACCLinear))
fprintf('Spec media para el discriminante lineal con Cancer: %f\n',mean(SpecLinear))
%validamos el modelo
numReps = 10
for j = 1:numReps
     cv = cvpartition(outputs,TypeCV,k);
     mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
    [RecallLinearTMP,SpecLinearTMP,PrecisionLinearTMP,NPVLinearTMP,ACCLinearTMP,F1ScoreLinearTMP, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 0);
    RecallLinearC(j) = mean(RecallLinearTMP);
    SpecLinearC(j) = mean(SpecLinearTMP);
    PrecisionLinearC(j) = mean(PrecisionLinearTMP);
    ACCLinearC(j) = mean(ACCLinearTMP);
    F1ScoreLinearC(j) = mean(F1ScoreLinearTMP);
end
%% muestra de resultados medios en test
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante lineal con Cancer: %f\n',mean(PrecisionLinearC))
fprintf('Recall media para el discriminante lineal con Cancer: %f\n',mean(RecallLinearC))
fprintf('ACC media para el discriminante lineal con Cancer: %f\n',mean(ACCLinearC))
fprintf('Spec media para el discriminante lineal con Cancer: %f\n',mean(SpecLinearC))

%% entrenamos con el discriminante cuadratico
typeDiscr = 'quadratic';
mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% mostramos datos de entrenamiento
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = predictResults(cv, inputs, outputs, mdls,1);
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para el discriminante cuadratico con Cancer: %f\n',mean(PrecisionQuadr))
fprintf('Recall media para el discriminante cuadratico con Cancer: %f\n',mean(RecallQuadr))
fprintf('ACC media para el discriminante cuadratico con Cancer: %f\n',mean(ACCQuadr))
fprintf('Spec media para el discriminante cuadratico con Cancer: %f\n',mean(SpecQuadr))
%% muestra de resultados medios
numReps = 10
for j = 1:numReps
     cv = cvpartition(outputs,TypeCV,k);
     mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
    [RecallQuadrTMP,SpecQuadrTMP,PrecisionQuadrTMP,NPVQuadrTMP,ACCQuadrTMP,F1ScoreQuadrTMP, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 0);
    RecallQuadrC(j) = mean(RecallQuadrTMP);
    SpecQuadrC(j) = mean(SpecQuadrTMP);
    PrecisionQuadrC(j) = mean(PrecisionQuadrTMP);
    ACCQuadrC(j) = mean(ACCQuadrTMP);
    F1ScoreQuadrC(j) = mean(F1ScoreQuadrTMP);
end
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante cuadratico con Cancer: %f\n',mean(PrecisionQuadrC))
fprintf('Recall media para el discriminante cuadratico con Cancer: %f\n',mean(RecallQuadrC))
fprintf('ACC media para el discriminante cuadratico con Cancer: %f\n',mean(ACCQuadrC))
fprintf('Spec media para el discriminante cuadratico con Cancer: %f\n',mean(SpecQuadrC))

%% Entrenar árboles
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);
% MaxNumSplits: número máximo de bifurcaciones
% MinLeafSize: número mínimo de observaciones para poder crear un nodo hoja
% MinParentSize: cada nodo de ramificación tiene al menos MinParentSize observaciones
mdls1 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,1,10,'on',inputsNames); % por defecto
mdls2 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,10,150,'on',inputsNames); % consigue resultados muy similares a mdls1 pero es más simple
mdls3 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,10,150,'on',inputsNames);

view(mdls1{1},'Mode','graph')
view(mdls2{1},'Mode','graph')
view(mdls3{1},'Mode','graph')

%% Muestra los resultados medios en entrenamiento
[RecallTree1TMP,SpecTree1TMP,PrecisionTree1TMP,NPVTree1TMP,ACCTree1TMP,F1ScoreTree1TMP, predictionTree1] = predictResults(cv, inputs, outputs, mdls1, 1);
% [RecallTree2TMP,SpecTree2TMP,PrecisionTree2TMP,NPVTree2TMP,ACCTree2TMP,F1ScoreTree2TMP, predictionTree2] = predictResults(cv, inputs, outputs, mdls, 1);
% [RecallTree3TMP,SpecTree3TMP,PrecisionTree3TMP,NPVTree3TMP,ACCTree3TMP,F1ScoreTree3TMP, predictionTree3] = predictResults(cv, inputs, outputs, mdls, 1);

% Arbol 1
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para árbol 1 con Cancer: %f\n',mean(PrecisionTree1TMP))
fprintf('Recall media para árbol 1 con Cancer: %f\n',mean(RecallTree1TMP))
fprintf('ACC media para árbol 1 con Cancer: %f\n',mean(ACCTree1TMP))
fprintf('Spec media para árbol 1 con Cancer: %f\n',mean(SpecTree1TMP))

%% Muestra los resultados medios en test
[RecallTree1TMP,SpecTree1TMP,PrecisionTree1TMP,NPVTree1TMP,ACCTree1TMP,F1ScoreTree1TMP, predictionTree1] = predictResults(cv, inputs, outputs, mdls1, 0);
% [RecallTree3TMP,SpecTree3TMP,PrecisionTree3TMP,NPVTree3TMP,ACCTree3TMP,F1ScoreTree3TMP, predictionTree3] = predictResults(cv, inputs, outputs, mdls, 0);

% Arbol 1
fprintf('\nDatos de test\n')
fprintf('Precision media para árbol 1 con Cancer: %f\n',mean(PrecisionTree1TMP))
fprintf('Recall media para árbol 1 con Cancer: %f\n',mean(RecallTree1TMP))
fprintf('ACC media para árbol 1 con Cancer: %f\n',mean(ACCTree1TMP))
fprintf('Spec media para árbol 1 con Cancer: %f\n',mean(SpecTree1TMP))

% Arbol 2
[RecallTree2TMP,SpecTree2TMP,PrecisionTree2TMP,NPVTree2TMP,ACCTree2TMP,F1ScoreTree2TMP, predictionTree2] = predictResults(cv, inputs, outputs, mdls2, 0);
fprintf('\nDatos de test\n')
fprintf('Precision media para árbol 2 con Cancer: %f\n',mean(PrecisionTree2TMP))
fprintf('Recall media para árbol 2 con Cancer: %f\n',mean(RecallTree2TMP))
fprintf('ACC media para árbol 2 con Cancer: %f\n',mean(ACCTree2TMP))
fprintf('Spec media para árbol 2 con Cancer: %f\n',mean(SpecTree2TMP))

%% Diferencias significativas entre modelos
muestras = [ACCLinearC;ACCQuadrC]';
etiquetas = ['linear';'quadra'];
[P] = testEstadistico(muestras,etiquetas,0.05);