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
% boxplot(inputs)

%% RNA
TypeCV = 'KFold';
k = 10;
% cv = cvpartition(outputs,TypeCV,k);

numReps = 3;
outputs = outputs';
inputs = inputs';
for j = 1:numReps
    cv = cvpartition(outputs,TypeCV,k);
    
    % Modelos GCE
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1GCE = trainingRNA(cv, inputs, outputs, 9, 'trainscg', 'crossentropy', 0.80, 0.20);
    mdls2GCE = trainingRNA(cv, inputs, outputs, 17, 'trainscg', 'crossentropy', 0.80, 0.20);
    mdls3GCE = trainingRNA(cv, inputs, outputs, 34, 'trainscg', 'crossentropy', 0.80, 0.20);
    mdlMatrixGCE = [mdls1GCE;mdls2GCE;mdls3GCE];
    
    % Modelos LM
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1LM = trainingRNA(cv, inputs, outputs, 9, 'trainlm', 'mse', 0.80, 0.20);
    mdls2LM = trainingRNA(cv, inputs, outputs, 17, 'trainlm', 'mse', 0.80, 0.20);
    mdls3LM = trainingRNA(cv, inputs, outputs, 34, 'trainlm', 'mse', 0.80, 0.20);
    mdlMatrixLM = [mdls1LM;mdls2LM;mdls3LM];
    
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrixGCE,1)
        [RecallRNATMP,SpecRNATMP,PrecisionRNATMP,NPVRNATMP,ACCRNATMP,F1ScoreRNATMP, predictionRNA] = predictResultsRNA(cv, inputs, outputs, mdlMatrixGCE(n,:), 0);
        RecallRNAGCE(numReps* (n-1) + j,:) = [RecallRNATMP];
        SpecRNAGCE(numReps* (n-1) + j,:) = [SpecRNATMP];
        PrecisionRNAGCE(numReps* (n-1) + j,:) = [PrecisionRNATMP];
        ACCRNAGCE(numReps* (n-1) + j,:) = [ACCRNATMP];
        F1ScoreRNAGCE(numReps* (n-1) + j,:) = [F1ScoreRNATMP];
    end
    for n = 1:size(mdlMatrixLM,1)
        [RecallRNATMP,SpecRNATMP,PrecisionRNATMP,NPVRNATMP,ACCRNATMP,F1ScoreRNATMP, predictionRNA] = predictResultsRNA(cv, inputs, outputs, mdlMatrixLM(n,:), 0);
        RecallRNALM(numReps* (n-1) + j,:) = [RecallRNATMP];
        SpecRNALM(numReps* (n-1) + j,:) = [SpecRNATMP];
        PrecisionRNALM(numReps* (n-1) + j,:) = [PrecisionRNATMP];
        ACCRNALM(numReps* (n-1) + j,:) = [ACCRNATMP];
        F1ScoreRNALM(numReps* (n-1) + j,:) = [F1ScoreRNATMP];
    end    
%     
end

%% Muestra los resultados medios en entrenamiento GCE
fprintf('\nENTRENAMIENTO')
%util para los print
kernels = ['RNAGCE1-9 ';'RNAGCE1-17';'RNAGCE1-34'];
for i = 1:size(mdlMatrixGCE,1)
    [RecallRNATMP,SpecRNATMP,PrecisionRNATMP,NPVRNATMP,ACCRNATMP,F1ScoreRNATMP, predictionRNA] = predictResultsRNA(cv, inputs, outputs, mdlMatrixGCE(i,:), 1);
    fprintf('\nDatos de entrenamiento de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(PrecisionRNATMP))
    fprintf('Recall media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(RecallRNATMP))
    fprintf('ACC media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(ACCRNATMP))
    fprintf('Spec media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(SpecRNATMP))
end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrixGCE,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(PrecisionRNAGCE(inicio:fin,:))))
    fprintf('Recall media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(RecallRNAGCE(inicio:fin,:))))
    fprintf('ACC media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(ACCRNAGCE(inicio:fin,:))))
    fprintf('Spec media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(SpecRNAGCE(inicio:fin,:))))
end

%% Muestra los resultados medios en entrenamiento LM
fprintf('\nENTRENAMIENTO')
%util para los print
kernels = ['RNALM1-9 ';'RNALM1-17';'RNALM1-34'];
for i = 1:size(mdlMatrixLM,1)
    [RecallRNATMP,SpecRNATMP,PrecisionRNATMP,NPVRNATMP,ACCRNATMP,F1ScoreRNATMP, predictionRNA] = predictResultsRNA(cv, inputs, outputs, mdlMatrixLM(i,:), 1);
    fprintf('\nDatos de entrenamiento de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(PrecisionRNATMP))
    fprintf('Recall media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(RecallRNATMP))
    fprintf('ACC media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(ACCRNATMP))
    fprintf('Spec media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(SpecRNATMP))
end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrixLM,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(PrecisionRNALM(inicio:fin,:))))
    fprintf('Recall media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(RecallRNALM(inicio:fin,:))))
    fprintf('ACC media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(ACCRNALM(inicio:fin,:))))
    fprintf('Spec media para RNA %s con Diabetes: %f\n',kernels(i,:),mean(mean(SpecRNALM(inicio:fin,:))))
end

%% Diferencias significativas entre modelos
%obtenemos las medias de las repeticiones 
ACCRNAGCE1_9 = mean(ACCRNAGCE(1:3,:));
ACCRNAGCE1_17 = mean(ACCRNAGCE(4:6,:));
ACCRNAGCE1_34 = mean(ACCRNAGCE(7:9,:));
ACCRNALM1_9 = mean(ACCRNALM(1:3,:));
ACCRNALM1_17 = mean(ACCRNALM(4:6,:));
ACCRNALM1_34 = mean(ACCRNALM(7:9,:));

load('resultadosDiabetesP3.mat')

% muestras = [ACCLinearC;ACCQuadrC;ACCTree(1,:);ACCTree(2,:);ACCTree(3,:);ACCSVMGauss ;ACCSVMPoly ;ACCSVMLinear ]';
% etiquetas = ['linear';'quadra';'tree_1';'tree_2';'tree_3';'SVMGau';'SVMPol';'SVMlin'];

muestras = [muestrasDiabetes'; ACCRNAGCE1_9; ACCRNAGCE1_17; ACCRNAGCE1_34; ACCRNALM1_9; ACCRNALM1_17; ACCRNALM1_34]';
etiquetas = ['linear    ';'quadra    ';'tree_1    ';'tree_2    ';'tree_3    ';'SVMGau    ';'SVMPol    ';'SVMlin    '; 'RNAGCE1-9 ';'RNAGCE1-17';'RNAGCE1-34'; 'RNALM1-9  ';'RNALM1-17 ';'RNALM1-34 '];
[P] = testEstadistico(muestras,etiquetas,0.05);


