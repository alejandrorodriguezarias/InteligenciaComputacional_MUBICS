%% limpiar  IDE
clear all
%% cargamos la base de datos de diabetes
loadHenon('Henon.mat');
data = x(1:1500);
dG1 = 10;
dG2 = 15;
[inputsG1, outputsG1] = slidingwindow(data,dG1);
[inputsG2, outputsG2] = slidingwindow(data,dG2);

%semilla aleatoria
rng('shuffle')

%% normalizaciï¿½n
[inputsG1,ps] = mapminmax(inputsG1, -1, 1);
[outputsG1,ps2] = mapminmax(outputsG1, -1, 1);

[inputsG2,ps] = mapminmax(inputsG2, -1, 1);
[outputsG2,ps2] = mapminmax(outputsG2, -1, 1);

%% RNA Ventana 1
TypeCV = 'KFold';
k = 10;
numReps = 3;
outputs = outputsG1';
inputs = inputsG1';
for j = 1:numReps
    for i=1:k
        cvTraining(i,:) = randperm(size(inputs,2),round(size(inputs,2)*0.9));
        indicesTot = randperm(size(inputs,2));
        % Diferencia entre conjunto de training y 
        % conjunto con todos los indice, obteniendo el conjunto de test
        cvTest(i,:) = setdiff(indicesTot,cvTraining(i,:)); 
    end    
    % Modelos GCE
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1GCE = trainingRRNA(cvTraining, inputs, outputs, 5, 'trainscg', 'mse', 0.80, 0.20);
    mdls2GCE = trainingRRNA(cvTraining, inputs, outputs, 10, 'trainscg', 'mse', 0.80, 0.20);
    mdls3GCE = trainingRRNA(cvTraining, inputs, outputs, 20, 'trainscg', 'mse', 0.80, 0.20);
    mdlMatrixGCE = [mdls1GCE;mdls2GCE;mdls3GCE];
    
    % Modelos LM
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1LM = trainingRRNA(cvTraining, inputs, outputs, 5, 'trainlm', 'mse', 0.80, 0.20);
    mdls2LM = trainingRRNA(cvTraining, inputs, outputs, 10, 'trainlm', 'mse', 0.80, 0.20);
    mdls3LM = trainingRRNA(cvTraining, inputs, outputs, 20, 'trainlm', 'mse', 0.80, 0.20);
    mdlMatrixLM = [mdls1LM;mdls2LM;mdls3LM];
    
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrixGCE,1)
        [ECMtmp] = calcularECMRNA(cvTraining, cvTest, inputs, outputs, mdlMatrixGCE(n,:), 0);
        ECMGCE(numReps* (n-1) + j,:) = [ECMtmp];
    end
    for n = 1:size(mdlMatrixLM,1)
        [ECMtmp] = calcularECMRNA(cvTraining, cvTest, inputs, outputs, mdlMatrixLM(n,:), 0);
        ECMLM(numReps* (n-1) + j,:) = [ECMtmp];
    end
end

ECMGCEG1 = ECMGCE;
ECMLMG1 = ECMLM;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%% RNA Ventana 2
TypeCV = 'KFold';
k = 10;
numReps = 3;
outputs = outputsG2';
inputs = inputsG2';
ECMLM = [];
ECMGCE = [];
cvTraining = [];
cvTest = [];
for j = 1:numReps
    for i=1:k
        cvTraining(i,:) = randperm(size(inputs,2),round(size(inputs,2)*0.9));
        indicesTot = randperm(size(inputs,2));
        % Diferencia entre conjunto de training y 
        % conjunto con todos los indice, obteniendo el conjunto de test
        cvTest(i,:) = setdiff(indicesTot,cvTraining(i,:)); 
    end    
    % Modelos GCE
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1GCE = trainingRRNA(cvTraining, inputs, outputs, 7, 'trainscg', 'mse', 0.80, 0.20);
    mdls2GCE = trainingRRNA(cvTraining, inputs, outputs, 15, 'trainscg', 'mse', 0.80, 0.20);
    mdls3GCE = trainingRRNA(cvTraining, inputs, outputs, 30, 'trainscg', 'mse', 0.80, 0.20);
    mdlMatrixGCE = [mdls1GCE;mdls2GCE;mdls3GCE];
    
    % Modelos LM
%   mdl = trainingRNA(cv, inputs, outputs, hiddenLayerSize, trainFCN, performFCN, trainRatio, valRatio);
    mdls1LM = trainingRRNA(cvTraining, inputs, outputs, 7, 'trainlm', 'mse', 0.80, 0.20);
    mdls2LM = trainingRRNA(cvTraining, inputs, outputs, 15, 'trainlm', 'mse', 0.80, 0.20);
    mdls3LM = trainingRRNA(cvTraining, inputs, outputs, 30, 'trainlm', 'mse', 0.80, 0.20);
    mdlMatrixLM = [mdls1LM;mdls2LM;mdls3LM];
    
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrixGCE,1)
        [ECMtmp] = calcularECMRNA(cvTraining, cvTest, inputs, outputs, mdlMatrixGCE(n,:), 0);
        ECMGCE(numReps* (n-1) + j,:) = [ECMtmp];
    end
    for n = 1:size(mdlMatrixLM,1)
        [ECMtmp] = calcularECMRNA(cvTraining, cvTest, inputs, outputs, mdlMatrixLM(n,:), 0);
        ECMLM(numReps* (n-1) + j,:) = [ECMtmp];
    end
end

ECMGCEG2 = ECMGCE;
ECMLMG2 = ECMLM;


%% Comparaciones

% Ventana 1
ECMRNAGCEG1_1 = mean(ECMGCEG1(1:3,:));
ECMRNAGCEG1_2 = mean(ECMGCEG1(4:6,:));
ECMRNAGCEG1_3 = mean(ECMGCEG1(7:9,:));
ECMRNALMG1_1 = mean(ECMLMG1(1:3,:));
ECMRNALMG1_2 = mean(ECMLMG1(4:6,:));
ECMRNALMG1_3 = mean(ECMLMG1(7:9,:));

% Ventana 2
ECMRNAGCEG2_1 = mean(ECMGCEG2(1:3,:));
ECMRNAGCEG2_2 = mean(ECMGCEG2(4:6,:));
ECMRNAGCEG2_3 = mean(ECMGCEG2(7:9,:));
ECMRNALMG2_1 = mean(ECMLMG2(1:3,:));
ECMRNALMG2_2 = mean(ECMLMG2(4:6,:));
ECMRNALMG2_3 = mean(ECMLMG2(7:9,:));

load('resultadosHenonG1.mat')
muestrasG1 = muestras;
load('resultadosHenonG2.mat')
muestrasG2 = muestras;

muestras = [muestrasG1'; muestrasG2'; ECMRNAGCEG1_1; ECMRNAGCEG1_2; ECMRNAGCEG1_3; ECMRNALMG1_1; ECMRNALMG1_2; ECMRNALMG1_3; 
            ECMRNAGCEG2_1; ECMRNAGCEG2_2; ECMRNAGCEG2_3; ECMRNALMG2_1; ECMRNALMG2_2; ECMRNALMG2_3]';
etiquetas = ['SVMGauG1    ';'SVMPolG1    ';'SVMlinG1    ';'SVMGauG2    ';'SVMPolG2    ';'SVMlinG2    ';
            'RNAGCE1-5G1 ';'RNAGCE1-10G1';'RNAGCE1-20G1'; 'RNALM1-5G1  ';'RNALM1-10G1 ';'RNALM1-20G1 ';
            'RNAGCE1-5G2 ';'RNAGCE1-10G2';'RNAGCE1-20G2'; 'RNALM1-5G2  ';'RNALM1-10G2 ';'RNALM1-20G2 '];
[P] = testEstadistico(muestras,etiquetas,0.05);
