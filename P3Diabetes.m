%% limpiar  IDE
clear all
%% cargamos la base de datos de diabetes
run loadDiabetes
inputs = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9, VarName10, VarName11, VarName12, VarName13, VarName14, VarName15, VarName16, VarName17, VarName18, VarName19];
outputs = VarName20;
%semilla aleatoria
rng('shuffle')

%% eliminar valores nulos (se podría rellenar con valores medios)
dataset = [inputs,outputs];
%filas con valores nulos
rows = any(isnan(inputs),2);
sum(rows);
%% eliminamos variables con valores constantes
inputs(:,1) = []; %1147 constantes
inputs(:,1) = []; %1057 constantes

%% normalizaciï¿½n
% inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
inputs = normalize(inputs);
% [inputs,ps] = mapminmax(inputs, -1, 1);
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
fprintf('\nValores para diabetes\n');
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para el discriminante lineal con diabetes: %f\n',mean(PrecisionLinear))
fprintf('Recall media para el discriminante lineal con diabetes: %f\n',mean(RecallLinear))
fprintf('ACC media para el discriminante lineal con diabetes: %f\n',mean(ACCLinear))
fprintf('Spec media para el discriminante lineal con diabetes: %f\n',mean(SpecLinear))
%% validamos el modelo
numReps = 10;
for j = 1:numReps
     cv = cvpartition(outputs,TypeCV,k);
     mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
    [RecallLinearTMP,SpecLinearTMP,PrecisionLinearTMP,NPVLinearTMP,ACCLinearTMP,F1ScoreLinearTMP, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 0);
    %cada posición del array final representará una media de los resultados
    %de un 10-fold
    RecallLinear(j) = mean(RecallLinearTMP);
    SpecLinear(j) = mean(SpecLinearTMP);
    PrecisionLinear(j) = mean(PrecisionLinearTMP);
    ACCLinear(j) = mean(ACCLinearTMP);
    F1ScoreLinear(j) = mean(F1ScoreLinearTMP);
end
%% muestra de resultados medios en test
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante lineal con diabetes: %f\n',mean(PrecisionLinear))
fprintf('Recall media para el discriminante lineal con diabetes: %f\n',mean(RecallLinear))
fprintf('ACC media para el discriminante lineal con diabetes: %f\n',mean(ACCLinear))
fprintf('Spec media para el discriminante lineal con diabetes: %f\n',mean(SpecLinear))

%% entrenamos con el discriminante cuadratico
typeDiscr = 'quadratic';
mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% mostramos datos de entrenamiento
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = predictResults(cv, inputs, outputs, mdls,1);
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para el discriminante cuadratico con diabetes: %f\n',mean(PrecisionQuadr))
fprintf('Recall media para el discriminante cuadratico con diabetes: %f\n',mean(RecallQuadr))
fprintf('ACC media para el discriminante cuadratico con diabetes: %f\n',mean(ACCQuadr))
fprintf('Spec media para el discriminante cuadratico con diabetes: %f\n',mean(SpecQuadr))
%% validamos el modelo
numReps = 10;
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
    [RecallQuadrTMP,SpecQuadrTMP,PrecisionQuadrTMP,NPVQuadrTMP,ACCQuadrTMP,F1ScoreQuadrTMP, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 0);
    %cada posición del array final representará una media de los resultados
    %de un 10-fold
    RecallQuadr(j) = mean(RecallQuadrTMP);
    SpecQuadr(j) = mean(SpecQuadrTMP);
    PrecisionQuadr(j) = mean(PrecisionQuadrTMP);
    ACCQuadr(j) = mean(ACCQuadrTMP);
    F1ScoreQuadr(j) = mean(F1ScoreQuadrTMP);
end
%% muestra de resultados medios en test
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante cuadratico con diabetes: %f\n',mean(PrecisionQuadr))
fprintf('Recall media para el discriminante cuadratico con diabetes: %f\n',mean(RecallQuadr))
fprintf('ACC media para el discriminante cuadratico con diabetes: %f\n',mean(ACCQuadr))
fprintf('Spec media para el discriminante cuadratico con diabetes: %f\n',mean(SpecQuadr))

%% Entrenar árboles
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);
%nombres para el display de variables en los arboles
inputsNames = {'V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19'};
numReps = 10;
for j = 1:numReps
    % MaxNumSplits: número máximo de bifurcaciones
    % MinLeafSize: número mínimo de observaciones para poder crear un nodo hoja
    % MinParentSize: cada nodo de ramificación tiene al menos MinParentSize observaciones
    cv = cvpartition(outputs,TypeCV,k);
    mdls1 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,1,10,'on',inputsNames,'gdi'); % por defecto
    mdls2 = trainingTree(cv,inputs,outputs,30,5,200,'on',inputsNames,'gdi'); 
    mdls3 = trainingTree(cv,inputs,outputs,10,50,100,'on',inputsNames,'twoing');
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    for n = 1:size(mdlMatrix,1)
        [RecallTreeTMP,SpecTreeTMP,PrecisionTreeTMP,NPVTreeTMP,ACCTreeTMP,F1ScoreTreeTMP, predictionTree] = predictResults(cv, inputs, outputs, mdlMatrix(n,:), 0);
        %cada posición del array final representará una media de los resultados
        %de un 10-fold; Cada fila corresponde a un arbol
        RecallTree(n,j) = mean(RecallTreeTMP);
        SpecTree(n,j) = mean(SpecTreeTMP);
        PrecisionTree(n,j) = mean(PrecisionTreeTMP);
        ACCTree(n,j) = mean(ACCTreeTMP);
        F1ScoreTree(n,j) = mean(F1ScoreTreeTMP);
    end
end

%% Muestra los resultados medios en entrenamiento
fprintf('\nENTRENAMIENTO')

for i = 1:size(mdlMatrix,1)
    [RecallTreeTMP,SpecTreeTMP,PrecisionTreeTMP,NPVTreeTMP,ACCTreeTMP,F1ScoreTreeTMP, predictionTree] = predictResults(cv, inputs, outputs, mdlMatrix(i,:), 1);
    fprintf('\nDatos de entrenamiento de árbol %d\n',i)
    fprintf('Precision media para árbol %d con diabetes: %f\n',i,mean(PrecisionTreeTMP))
    fprintf('Recall media para árbol %d con diabetes: %f\n',i,mean(RecallTreeTMP))
    fprintf('ACC media para árbol %d con diabetes: %f\n',i,mean(ACCTreeTMP))
    fprintf('Spec media para árbol %d con diabetes: %f\n',i,mean(SpecTreeTMP))
end


%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    fprintf('\nDatos de test de árbol %d\n',i)
    fprintf('Precision media para árbol %d con diabetes: %f\n',i,mean(PrecisionTree(i,:)))
    fprintf('Recall media para árbol %d con diabetes: %f\n',i,mean(RecallTree(i,:)))
    fprintf('ACC media para árbol %d con diabetes: %f\n',i,mean(ACCTree(i,:)))
    fprintf('Spec media para árbol %d con diabetes: %f\n',i,mean(SpecTree(i,:)))
end
%mostrar ejemplos de arboles
view(mdls1{1},'Mode','graph')
view(mdls2{1},'Mode','graph')
view(mdls3{1},'Mode','graph')

%% Entrenar SVM
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);

numReps = 3;
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    %%%%%%%% USAR PARAMETRO C
    %kernel gaussiano con sigma 2
    mdls1 = trainingSVM(cv,inputs,outputs,1,'gaussian',2);
    %kernel polinomial de grado 2
    mdls2 = trainingSVM(cv,inputs,outputs,1,'polynomial',2);
    %kernel lineal //se podría substituir por polinomial de grado 1
    mdls3 = trainingSVM(cv,inputs,outputs,1,'linear',0);
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrix,1)
        [RecallSVMTMP,SpecSVMTMP,PrecisionSVMTMP,NPVSVMTMP,ACCSVMTMP,F1ScoreSVMTMP, predictionSVM] = predictResults(cv, inputs, outputs, mdlMatrix(n,:), 0);
        RecallSVM(numReps* (n-1) + j,:) = [RecallSVMTMP];
        SpecSVM(numReps* (n-1) + j,:) = [SpecSVMTMP];
        PrecisionSVM(numReps* (n-1) + j,:) = [PrecisionSVMTMP];
        ACCSVM(numReps* (n-1) + j,:) = [ACCSVMTMP];
        F1ScoreSVM(numReps* (n-1) + j,:) = [F1ScoreSVMTMP];
    end
end


%% Muestra los resultados medios en entrenamiento
fprintf('\nENTRENAMIENTO')
%util para los print
kernels = ['gaussiano ';'polinomico';'linear    '];
for i = 1:size(mdlMatrix,1)
    [RecallSVMTMP,SpecSVMTMP,PrecisionSVMTMP,NPVSVMTMP,ACCSVMTMP,F1ScoreSVMTMP, predictionSVM] = predictResults(cv, inputs, outputs, mdlMatrix(i,:), 1);
    fprintf('\nDatos de entrenamiento de SVM %s\n',kernels(i,:))
    fprintf('Precision media para SVM %s con diabetes: %f\n',kernels(i,:),mean(PrecisionSVMTMP))
    fprintf('Recall media para SVM %s con diabetes: %f\n',kernels(i,:),mean(RecallSVMTMP))
    fprintf('ACC media para SVM %s con diabetes: %f\n',kernels(i,:),mean(ACCSVMTMP))
    fprintf('Spec media para SVM %s con diabetes: %f\n',kernels(i,:),mean(SpecSVMTMP))
end


%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de SVM %s\n',kernels(i,:))
    fprintf('Precision media para SVM %s con diabetes: %f\n',kernels(i,:),mean(mean(PrecisionSVM(inicio:fin,:))))
    fprintf('Recall media para SVM %s con diabetes: %f\n',kernels(i,:),mean(mean(RecallSVM(inicio:fin,:))))
    fprintf('ACC media para SVM %s con diabetes: %f\n',kernels(i,:),mean(mean(ACCSVM(inicio:fin,:))))
    fprintf('Spec media para SVM %s con diabetes: %f\n',kernels(i,:),mean(mean(SpecSVM(inicio:fin,:))))
end
%% Diferencias significativas entre modelos
%obtenemos las medias de las repeticiones para cada kernel
ACCSVMGauss = mean(ACCSVM(1:3,:));
ACCSVMPoly = mean(ACCSVM(4:6,:));
ACCSVMLinear = mean(ACCSVM(7:9,:));
muestras = [ACCLinear;ACCQuadr;ACCTree(1,:);ACCTree(2,:);ACCTree(3,:);ACCSVMGauss ;ACCSVMPoly ;ACCSVMLinear ]';
etiquetas = ['linear';'quadra';'tree_1';'tree_2';'tree_3';'SVMGau';'SVMPol';'SVMlin'];
[P] = testEstadistico(muestras,etiquetas,0.05);