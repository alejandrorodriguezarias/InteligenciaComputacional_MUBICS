%% limpiar  
clear all
%% cargamos la base de datos de cancer
run loadCancer
inputs = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9];
inputsNames = {'Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'};
outputs = ClaseCancer;

rng('shuffle')

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
outputs = newmatrix';
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
numReps = 10;
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
numReps = 10;
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


numReps = 10;
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    mdls1 = trainingTree(cv,inputs,outputs,min(cv.TrainSize)-1,1,10,'on',inputsNames,'gdi'); % por defecto
    mdls2 = trainingTree(cv,inputs,outputs,20,5,30,'on',inputsNames,'gdi'); 
    mdls3 = trainingTree(cv,inputs,outputs,3,20,300,'on',inputsNames,'twoing');
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    for n = 1:size(mdlMatrix,1)
        [RecallTreeTMP,SpecTreeTMP,PrecisionTreeTMP,NPVTreeTMP,ACCTreeTMP,F1ScoreTreeTMP, predictionTree] = predictResults(cv, inputs, outputs, mdlMatrix(n,:), 0);
        
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
    fprintf('Precision media para árbol %d con Cancer: %f\n',i,mean(PrecisionTreeTMP))
    fprintf('Recall media para árbol %d con Cancer: %f\n',i,mean(RecallTreeTMP))
    fprintf('ACC media para árbol %d con Cancer: %f\n',i,mean(ACCTreeTMP))
    fprintf('Spec media para árbol %d con Cancer: %f\n',i,mean(SpecTreeTMP))
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
muestras = [ACCLinearC;ACCQuadrC;ACCTree(1,:);ACCTree(2,:);ACCTree(3,:)]';
etiquetas = ['linear';'quadra';'tree_1';'tree_2';'tree_3'];
[P] = testEstadistico(muestras,etiquetas,0.05);