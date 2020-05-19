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
%eliminaciï¿½n de valores nulos
dataset(rows,:) = [];
inputs = dataset(:,1:9);
outputs = dataset(:,10);
%renombramos las etiquetas de clase
newmatrix(outputs == 2) = 0; % benigno
newmatrix(outputs == 4) = 1; % maligno
outputs = newmatrix';
% eliminamos las variables correlacionadas
% corrcoef(inputs);
% inputs(:,2) = [];
%% normalizaciï¿½n
% inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
inputs = normalize(inputs);

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
    mdls1GCE = trainingRNA(cv, inputs, outputs, 4, 'trainscg', 'crossentropy', 0.80, 0.20);
    mdls2GCE = trainingRNA(cv, inputs, outputs, 8, 'trainscg', 'crossentropy', 0.80, 0.20);
    mdls3GCE = trainingRNA(cv, inputs, outputs, 16, 'trainscg', 'crossentropy', 0.80, 0.20);
    mdlMatrixGCE = [mdls1GCE;mdls2GCE;mdls3GCE];
    
    % Modelos LM
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1LM = trainingRNA(cv, inputs, outputs, 4, 'trainlm', 'mse', 0.80, 0.20);
    mdls2LM = trainingRNA(cv, inputs, outputs, 8, 'trainlm', 'mse', 0.80, 0.20);
    mdls3LM = trainingRNA(cv, inputs, outputs, 16, 'trainlm', 'mse', 0.80, 0.20);
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
kernels = ['RNAGCE1-4 ';'RNAGCE1-8 ';'RNAGCE1-16'];
for i = 1:size(mdlMatrixGCE,1)
    [RecallRNATMP,SpecRNATMP,PrecisionRNATMP,NPVRNATMP,ACCRNATMP,F1ScoreRNATMP, predictionRNA] = predictResultsRNA(cv, inputs, outputs, mdlMatrixGCE(i,:), 1);
    fprintf('\nDatos de entrenamiento de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Cancer: %f\n',kernels(i,:),mean(PrecisionRNATMP))
    fprintf('Recall media para RNA %s con Cancer: %f\n',kernels(i,:),mean(RecallRNATMP))
    fprintf('ACC media para RNA %s con Cancer: %f\n',kernels(i,:),mean(ACCRNATMP))
    fprintf('Spec media para RNA %s con Cancer: %f\n',kernels(i,:),mean(SpecRNATMP))
end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrixGCE,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(PrecisionRNAGCE(inicio:fin,:))))
    fprintf('Recall media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(RecallRNAGCE(inicio:fin,:))))
    fprintf('ACC media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(ACCRNAGCE(inicio:fin,:))))
    fprintf('Spec media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(SpecRNAGCE(inicio:fin,:))))
end

%% Muestra los resultados medios en entrenamiento LM
fprintf('\nENTRENAMIENTO')
%util para los print
kernels = ['RNALM1-4 ';'RNALM1-8 ';'RNALM1-16'];
for i = 1:size(mdlMatrixLM,1)
    [RecallRNATMP,SpecRNATMP,PrecisionRNATMP,NPVRNATMP,ACCRNATMP,F1ScoreRNATMP, predictionRNA] = predictResultsRNA(cv, inputs, outputs, mdlMatrixLM(i,:), 1);
    fprintf('\nDatos de entrenamiento de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Cancer: %f\n',kernels(i,:),mean(PrecisionRNATMP))
    fprintf('Recall media para RNA %s con Cancer: %f\n',kernels(i,:),mean(RecallRNATMP))
    fprintf('ACC media para RNA %s con Cancer: %f\n',kernels(i,:),mean(ACCRNATMP))
    fprintf('Spec media para RNA %s con Cancer: %f\n',kernels(i,:),mean(SpecRNATMP))
end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrixLM,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de RNA %s\n',kernels(i,:))
    fprintf('Precision media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(PrecisionRNALM(inicio:fin,:))))
    fprintf('Recall media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(RecallRNALM(inicio:fin,:))))
    fprintf('ACC media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(ACCRNALM(inicio:fin,:))))
    fprintf('Spec media para RNA %s con Cancer: %f\n',kernels(i,:),mean(mean(SpecRNALM(inicio:fin,:))))
end

%% Diferencias significativas entre modelos
%obtenemos las medias de las repeticiones 
ACCRNAGCE1_4 = mean(ACCRNAGCE(1:3,:));
ACCRNAGCE1_8 = mean(ACCRNAGCE(4:6,:));
ACCRNAGCE1_16 = mean(ACCRNAGCE(7:9,:));
ACCRNALM1_4 = mean(ACCRNALM(1:3,:));
ACCRNALM1_8 = mean(ACCRNALM(4:6,:));
ACCRNALM1_16 = mean(ACCRNALM(7:9,:));

load('resultadosCancerP3.mat')

% muestras = [ACCLinearC;ACCQuadrC;ACCTree(1,:);ACCTree(2,:);ACCTree(3,:);ACCSVMGauss ;ACCSVMPoly ;ACCSVMLinear ]';
% etiquetas = ['linear';'quadra';'tree_1';'tree_2';'tree_3';'SVMGau';'SVMPol';'SVMlin'];

muestras = [muestrasCancer'; ACCRNAGCE1_4; ACCRNAGCE1_8; ACCRNAGCE1_16; ACCRNALM1_4; ACCRNALM1_8; ACCRNALM1_16]';
etiquetas = ['linear    ';'quadra    ';'tree_1    ';'tree_2    ';'tree_3    ';'SVMGau    ';'SVMPol    ';'SVMlin    '; 'RNAGCE1-4 ';'RNAGCE1-8 ';'RNAGCE1-16'; 'RNALM1-4  ';'RNALM1-8  ';'RNALM1-16 '];
[P] = testEstadistico(muestras,etiquetas,0.05);



