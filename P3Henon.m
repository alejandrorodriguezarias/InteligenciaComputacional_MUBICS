%% limpiar  IDE
clear all
%% cargamos la base de datos de diabetes
loadHenon('Henon.mat');
data = x(1:1500);

d = 10;
[inputs, outputs] = slidingwindow(data,d);

%semilla aleatoria
rng('shuffle')

%% normalizaciï¿½n

[inputs,ps] = mapminmax(inputs, -1, 1);

[outputs,ps2] = mapminmax(outputs, -1, 1);

boxplot(inputs)

%% Entrenar SVM
TypeCV = 'KFold';
k = 10;

numReps = 3;

for j = 1:numReps
    for i=1:k
        cvTraining(i,:) = randperm(1490,1341);
        indicesTot = randperm(1490);
        % Diferencia entre conjunto de training y 
        % conjunto con todos los indice, obteniendo el conjunto de test
        cvTest(i,:) = setdiff(indicesTot,cvTraining(i,:)); 
        
    end

    %kernel gaussiano con sigma 2
    mdls1 = trainingRSVM(cvTraining,inputs,outputs,1,'gaussian',2);
    %kernel polinomial de grado 2
    mdls2 = trainingRSVM(cvTraining,inputs,outputs,1,'polynomial',2);
    %kernel lineal //se podría substituir por polinomial de grado 1
    mdls3 = trainingRSVM(cvTraining,inputs,outputs,1,'linear',0);
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrix,1)
        [ECMtmp] = calcularECM(cvTraining, cvTest, inputs, outputs, mdlMatrix(n,:), 0);
        
        ECM(numReps* (n-1) + j,:) = [ECMtmp];
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