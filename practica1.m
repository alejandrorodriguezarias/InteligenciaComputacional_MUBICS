%% limpiar  
clear all

%% cargamos la base de datos de Iris
run loadIris
inputs = [VarName1, VarName2, VarName3, VarName4];
outputs = Irissetosa;
%% cargamos la base de datos de cancer
%run loadIris
%inputs = [VarName1, VarName2, VarName3, VarName4];
%outputs = Irissetosa;
%% normalización
inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
%% 10-fold como particiï¿½n del conjunto de datos
typeDiscr = 'linear';
TypeCV = 'KFold';
k = 10;
cv = cvpartition(outputs,TypeCV,k);
%% entrenamos con el discriminante lineal
[RecallLinear,SpecLinear,PrecisionLinear,NPVLinear,ACCLinear,F1ScoreLinear, predictionLinear] = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% muestra de resultados medios
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(PrecisionLinear)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(RecallLinear)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACCLinear)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(SpecLinear)))


%% entrenamos con el discriminante cuadratico
typeDiscr = 'quadratic';
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% muestra de resultados medios
fprintf('Precision media para el discriminante cuadratico con Iris: %f\n',mean(mean(PrecisionQuadr)))
fprintf('Recall media para el discriminante cuadratico con Iris: %f\n',mean(mean(RecallQuadr)))
fprintf('ACC media para el discriminante cuadratico con Iris: %f\n',mean(mean(ACCQuadr)))
fprintf('Spec media para el discriminante cuadratico con Iris : %f\n',mean(mean(SpecQuadr)))

%% Curva ROC
%[X,Y,T,AUC] = perfcurve(outputs{cv.test(1)},prediction{1},'virginica');
%plot(X,Y)

%% Diferencias significativas entre modelos
ACCMeanLinear = mean(ACCLinear,2); % para cada modelo calculamos la ACC media para las tres clases de flor
ACCMeanQuadr = mean(ACCQuadr,2);

muestras = [ACCMeanLinear,ACCMeanQuadr];
etiquetas = ['linear';'quadra'];
[P] = testEstadistico(muestras,etiquetas,0.05)

