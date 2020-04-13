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

boxplot(inputsG1)
boxplot(inputsG2)

%% Entrenar SVM con Grupo 1
TypeCV = 'KFold';
k = 10;
numReps = 3;
for j = 1:numReps
    for i=1:k
        cvTraining(i,:) = randperm(size(inputsG1,1),round(size(inputsG1,1)*0.9));
        indicesTot = randperm(size(inputsG1,1));
        % Diferencia entre conjunto de training y 
        % conjunto con todos los indice, obteniendo el conjunto de test
        cvTest(i,:) = setdiff(indicesTot,cvTraining(i,:)); 
    end
    %kernel gaussiano con sigma 2
    mdls1 = trainingRSVM(cvTraining,inputsG1,outputsG1,1,'gaussian',2);
    %kernel polinomial de grado 2
    mdls2 = trainingRSVM(cvTraining,inputsG1,outputsG1,1,'polynomial',3);
    %kernel lineal //se podría substituir por polinomial de grado 1
    mdls3 = trainingRSVM(cvTraining,inputsG1,outputsG1,1,'linear',0);
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrix,1)
        [ECMtmp] = calcularECM(cvTraining, cvTest, inputsG1, outputsG1, mdlMatrix(n,:), 0);
        ECMG1(numReps* (n-1) + j,:) = [ECMtmp];
    end
end

%% Muestra los resultados medios en entrenamiento
fprintf('\nENTRENAMIENTO')
%util para los print
kernels = ['gaussiano ';'polinomico';'linear    '];
for i = 1:size(mdlMatrix,1)
    [ECMtmp2] = calcularECM(cvTraining, cvTest, inputsG1, outputsG1, mdlMatrix(i,:), 1);
    fprintf('\nDatos de entrenamiento de Henon %s\n',kernels(i,:))
    fprintf('ECMG1 para SVM %s con Henon: %f\n',kernels(i,:),mean(ECMtmp2))

end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de SVM %s\n',kernels(i,:))
    fprintf('ECMG1 para SVM %s con Henon: %f\n',kernels(i,:),mean(mean(ECMG1(inicio:fin,:))))

end



%% Entrenar SVM con Grupo 2
TypeCV = 'KFold';
k = 10;
numReps = 3;
cvTraining = [];
cvTest = [];
for j = 1:numReps
    for i=1:k
        cvTraining(i,:) = randperm(size(inputsG2,1),round(size(inputsG2,1)*0.9));
        indicesTot = randperm(size(inputsG2,1));
        % Diferencia entre conjunto de training y 
        % conjunto con todos los indice, obteniendo el conjunto de test
        cvTest(i,:) = setdiff(indicesTot,cvTraining(i,:)); 
    end
    %kernel gaussiano con sigma 2
    mdls1 = trainingRSVM(cvTraining,inputsG2,outputsG2,1,'gaussian',2);
    %kernel polinomial de grado 2
    mdls2 = trainingRSVM(cvTraining,inputsG2,outputsG2,1,'polynomial',3);
    %kernel lineal //se podría substituir por polinomial de grado 1
    mdls3 = trainingRSVM(cvTraining,inputsG2,outputsG2,1,'linear',0);
    
    mdlMatrix = [mdls1;mdls2;mdls3];
    %obtenemos una matriz de resultados ordenada por filas
    % 1º-3º filas para gaussiana
    % 4º - 6º filas para polinomial
    % 7º - 9º filas para linear
    for n = 1:size(mdlMatrix,1)
        [ECMtmp] = calcularECM(cvTraining, cvTest, inputsG2, outputsG2, mdlMatrix(n,:), 0);
        ECMG2(numReps* (n-1) + j,:) = [ECMtmp];
    end
end

%% Muestra los resultados medios en entrenamiento
fprintf('\nENTRENAMIENTO')
%util para los print
kernels = ['gaussiano ';'polinomico';'linear    '];
for i = 1:size(mdlMatrix,1)
    [ECMtmp2] = calcularECM(cvTraining, cvTest, inputsG2, outputsG2, mdlMatrix(i,:), 1);
    fprintf('\nDatos de entrenamiento de Henon %s\n',kernels(i,:))
    fprintf('ECMG2 para SVM %s con Henon: %f\n',kernels(i,:),mean(ECMtmp2))

end

%% Muestra los resultados medios del test
fprintf('\nTEST')
for i = 1:size(mdlMatrix,1)
    %obtenemos las medias de las repeticiones para cada kernel
    inicio = (i-1)*3+1;
    fin = inicio +(numReps-1);
    fprintf('\nDatos de test de SVM %s\n',kernels(i,:))
    fprintf('ECMG2 para SVM %s con Henon: %f\n',kernels(i,:),mean(mean(ECMG2(inicio:fin,:))))

end


%% Diferencias significativas entre modelos
%obtenemos las medias de las repeticiones para cada kernel
ECMSVMGaussG1 = mean(ECMG1(1:3,:));
ECMSVMPolyG1 = mean(ECMG1(4:6,:));
ECMSVMLinearG1 = mean(ECMG1(7:9,:));
ECMSVMGaussG2 = mean(ECMG2(1:3,:));
ECMSVMPolyG2 = mean(ECMG2(4:6,:));
ECMSVMLinearG2 = mean(ECMG2(7:9,:));
muestras = [ECMSVMGaussG1; ECMSVMPolyG1; ECMSVMLinearG1;ECMSVMGaussG2; ECMSVMPolyG2; ECMSVMLinearG2]';
etiquetas = ['SVMGauG1';'SVMPolG1';'SVMlinG1';'SVMGauG2';'SVMPolG2';'SVMlinG2'];
[P] = testEstadistico(muestras,etiquetas,0.05);

muestras = [ECMSVMGaussG1; ECMSVMPolyG1; ECMSVMLinearG1]';
etiquetas = ['SVMGauG1';'SVMPolG1';'SVMlinG1'];
[P] = testEstadistico(muestras,etiquetas,0.05);

muestras = [ECMSVMGaussG2; ECMSVMPolyG2; ECMSVMLinearG2]';
etiquetas = ['SVMGauG2';'SVMPolG2';'SVMlinG2'];
[P] = testEstadistico(muestras,etiquetas,0.05);