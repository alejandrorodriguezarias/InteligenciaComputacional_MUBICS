%% limpiar  IDE
clear all
%% cargamos la base de datos de cancer
run loadCancer
inputs = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9];
inputsNames = {'Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'};
outputs = ClaseCancer;
%semilla aleatoria
rng('shuffle')

%% eliminar valores nulos
dataset = [inputs,outputs];
%filas con valores nulos
rows = any(isnan(inputs),2);
%eliminaci�n de valores nulos
dataset(rows,:) = [];
inputs = dataset(:,1:9);
outputs = dataset(:,10);
%renombramos las etiquetas de clase
newmatrix(outputs == 2) = {'benigno'};
newmatrix(outputs == 4) = {'maligno'};
outputs = newmatrix';
% eliminamos las variables correlacionadas
corrcoef(inputs);
inputs(:,2) = [];
%% normalizaci�n
% inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
inputs = normalize(inputs);
% [inputs,ps] = mapminmax(inputs, -1, 1);
% boxplot(inputs)
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

%% validamos el modelo
numReps = 10;
for j = 1:numReps
     cv = cvpartition(outputs,TypeCV,k);
     mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
    [RecallLinearTMP,SpecLinearTMP,PrecisionLinearTMP,NPVLinearTMP,ACCLinearTMP,F1ScoreLinearTMP, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 0);
    %cada posici�n del array final representar� una media de los resultados
    %de un 10-fold
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
%% validamos el modelo
numReps = 10;
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
    [RecallQuadrTMP,SpecQuadrTMP,PrecisionQuadrTMP,NPVQuadrTMP,ACCQuadrTMP,F1ScoreQuadrTMP, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 0);
    %cada posici�n del array final representar� una media de los resultados
    %de un 10-fold
    RecallQuadrC(j) = mean(RecallQuadrTMP);
    SpecQuadrC(j) = mean(SpecQuadrTMP);
    PrecisionQuadrC(j) = mean(PrecisionQuadrTMP);
    ACCQuadrC(j) = mean(ACCQuadrTMP);
    F1ScoreQuadrC(j) = mean(F1ScoreQuadrTMP);
end
%% muestra de resultados medios en test
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante cuadratico con Cancer: %f\n',mean(PrecisionQuadrC))
fprintf('Recall media para el discriminante cuadratico con Cancer: %f\n',mean(RecallQuadrC))
fprintf('ACC media para el discriminante cuadratico con Cancer: %f\n',mean(ACCQuadrC))
fprintf('Spec media para el discriminante cuadratico con Cancer: %f\n',mean(SpecQuadrC))

%% Entrenar �rboles
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);

numReps = 10;
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    % MaxNumSplits: n�mero m�ximo de bifurcaciones
    % MinLeafSize: n�mero m�nimo de observaciones para poder crear un nodo hoja
    % MinParentSize: cada nodo de ramificaci�n tiene al menos MinParentSize observaciones
    mdls1 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,1,10,'on',inputsNames,'gdi'); % por defecto
    mdls2 = trainingTree(cv,inputs,outputs,20,5,30,'on',inputsNames,'gdi'); %arbol medio
    mdls3 = trainingTree(cv,inputs,outputs,3,20,300,'on',inputsNames,'twoing'); %arbol peque�o
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    for n = 1:size(mdlMatrix,1)
        [RecallTreeTMP,SpecTreeTMP,PrecisionTreeTMP,NPVTreeTMP,ACCTreeTMP,F1ScoreTreeTMP, predictionTree] = predictResults(cv, inputs, outputs, mdlMatrix(n,:), 0);
        %cada posici�n del array final representar� una media de los resultados
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
    fprintf('\nDatos de entrenamiento de �rbol %d\n',i)
    fprintf('Precision media para �rbol %d con Cancer: %f\n',i,mean(PrecisionTreeTMP))
    fprintf('Recall media para �rbol %d con Cancer: %f\n',i,mean(RecallTreeTMP))
    fprintf('ACC media para �rbol %d con Cancer: %f\n',i,mean(ACCTreeTMP))
    fprintf('Spec media para �rbol %d con Cancer: %f\n',i,mean(SpecTreeTMP))
end


%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    fprintf('\nDatos de test de �rbol %d\n',i)
    fprintf('Precision media para �rbol %d con Cancer: %f\n',i,mean(PrecisionTree(i,:)))
    fprintf('Recall media para �rbol %d con Cancer: %f\n',i,mean(RecallTree(i,:)))
    fprintf('ACC media para �rbol %d con Cancer: %f\n',i,mean(ACCTree(i,:)))
    fprintf('Spec media para �rbol %d con Cancer: %f\n',i,mean(SpecTree(i,:)))
end
%display de arboles
view(mdls1{1},'Mode','graph')
view(mdls2{1},'Mode','graph')
view(mdls3{1},'Mode','graph')

%% Entrenar SVM
TypeCV = 'KFold';
k = 10;
% cv = cvpartition(outputs,TypeCV,k);

numReps = 3;
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    %kernel gaussiano con sigma 1
    mdls1 = trainingSVM(cv,inputs,outputs,3,'gaussian',1);
    %kernel polinomial de grado 2
    mdls2 = trainingSVM(cv,inputs,outputs,0.5,'polynomial',2);
    %kernel lineal //se podr�a substituir por polinomial de grado 1
    mdls3 = trainingSVM(cv,inputs,outputs,0.5,'linear',0);
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    
    %obtenemos una matriz de resultados ordenada por filas
    % 1�-3� filas para gaussiana
    % 4� - 6� filas para polinomial
    % 7� - 9� filas para linear
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
kernels = ['gaussiano ';'polinomico';'linear    ']
for i = 1:size(mdlMatrix,1)
    [RecallSVMTMP,SpecSVMTMP,PrecisionSVMTMP,NPVSVMTMP,ACCSVMTMP,F1ScoreSVMTMP, predictionSVM] = predictResults(cv, inputs, outputs, mdlMatrix(i,:), 1);
    fprintf('\nDatos de entrenamiento de SVM %s\n',kernels(i,:))
    fprintf('Precision media para SVM %s con Cancer: %f\n',kernels(i,:),mean(PrecisionSVMTMP))
    fprintf('Recall media para SVM %s con Cancer: %f\n',kernels(i,:),mean(RecallSVMTMP))
    fprintf('ACC media para SVM %s con Cancer: %f\n',kernels(i,:),mean(ACCSVMTMP))
    fprintf('Spec media para SVM %s con Cancer: %f\n',kernels(i,:),mean(SpecSVMTMP))
end


%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de SVM %s\n',kernels(i,:))
    fprintf('Precision media para SVM %s con Cancer: %f\n',kernels(i,:),mean(mean(PrecisionSVM(inicio:fin,:))))
    fprintf('Recall media para SVM %s con Cancer: %f\n',kernels(i,:),mean(mean(RecallSVM(inicio:fin,:))))
    fprintf('ACC media para SVM %s con Cancer: %f\n',kernels(i,:),mean(mean(ACCSVM(inicio:fin,:))))
    fprintf('Spec media para SVM %s con Cancer: %f\n',kernels(i,:),mean(mean(SpecSVM(inicio:fin,:))))
end
%% Diferencias significativas entre modelos
%obtenemos las medias de las repeticiones para cada kernel
ACCSVMGauss = mean(ACCSVM(1:3,:));
ACCSVMPoly = mean(ACCSVM(4:6,:));
ACCSVMLinear = mean(ACCSVM(7:9,:));
muestrasCancer = [ACCLinearC;ACCQuadrC;ACCTree(1,:);ACCTree(2,:);ACCTree(3,:);ACCSVMGauss ;ACCSVMPoly ;ACCSVMLinear ]';
etiquetas = ['linear';'quadra';'tree_1';'tree_2';'tree_3';'SVMGau';'SVMPol';'SVMlin'];
[P] = testEstadistico(muestrasCancer,etiquetas,0.05);

save('resultadosCancerP3', 'muestrasCancer')
