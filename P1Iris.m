%% limpiar  
clear all
%cambiar la semilla 
rng('shuffle')
%% cargamos la base de datos de Iris
run loadIris
%aleatorizamos los valores
inputs = [VarName1, VarName2, VarName3, VarName4];
rdnIndx = randperm(size(inputs,1),size(inputs,1));
inputs = inputs(rdnIndx,:);
outputs = Irissetosa(rdnIndx,:);

%% normalizaci�n
% inputs = (inputs - mean(mean(inputs)))/std(std(inputs));
inputs = normalize(inputs);
boxplot(inputs);

%corrcoef(inputs);
% eliminamos la variable anchura del petalo por estar altamente 
% correlacionada con la longitud del petado (0,96)
%inputs(:,4) = [];
%corrcoef(inputs);
%% 10-fold como partici�n del conjunto de datos
typeDiscr = 'linear';
TypeCV = 'LeaveOut';
cv = cvpartition(outputs,TypeCV);
%% entrenamos con el discriminante lineal
mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% mostramos datos de entrenamiento
[RecallLinear,SpecLinear,PrecisionLinear,NPVLinear,ACCLinear,F1ScoreLinear, predictionLinear] = predictResults(cv, inputs, outputs, mdls, 1);
fprintf('Datos de entrenamiento\n')
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(PrecisionLinear)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(RecallLinear)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACCLinear)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(SpecLinear)))

%% muestra de resultados de test
[RecallLinear,SpecLinear,PrecisionLinear,NPVLinear,ACCLinear,F1ScoreLinear, predictionLinear] = predictResultsLOO(cv, inputs, outputs, mdls);
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante lineal con Iris: %f\n',mean(mean(PrecisionLinear)))
fprintf('Recall media para el discriminante lineal con Iris: %f\n',mean(mean(RecallLinear)))
fprintf('ACC media para el discriminante lineal con Iris: %f\n',mean(mean(ACCLinear)))
fprintf('Spec media para el discriminante lineal con Iris : %f\n',mean(mean(SpecLinear)))


%% entrenamos con el discriminante cuadratico
typeDiscr = 'quadratic';
mdls = trainingDiscr(typeDiscr, cv, inputs, outputs);
%% mostramos datos de entrenamiento
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = predictResults(cv, inputs, outputs, mdls,1);
fprintf('\nDatos de entrenamiento\n')
fprintf('Precision media para el discriminante cuadratico con Iris: %f\n',mean(mean(PrecisionQuadr)))
fprintf('Recall media para el discriminante cuadratico con Iris: %f\n',mean(mean(RecallQuadr)))
fprintf('ACC media para el discriminante cuadratico con Iris: %f\n',mean(mean(ACCQuadr)))
fprintf('Spec media para el discriminante cuadratico con Iris : %f\n',mean(mean(SpecQuadr)))

%% muestra de resultados medios
[RecallQuadr,SpecQuadr,PrecisionQuadr,NPVQuadr,ACCQuadr,F1ScoreQuadr, predictionQuadr] = predictResults(cv, inputs, outputs, mdls,0);
fprintf('\nDatos de test\n')
fprintf('Precision media para el discriminante cuadratico con Iris: %f\n',mean(mean(PrecisionQuadr)))
fprintf('Recall media para el discriminante cuadratico con Iris: %f\n',mean(mean(RecallQuadr)))
fprintf('ACC media para el discriminante cuadratico con Iris: %f\n',mean(mean(ACCQuadr)))
fprintf('Spec media para el discriminante cuadratico con Iris : %f\n',mean(mean(SpecQuadr)))

%% Curva ROC
plot((1-SpecQuadr),RecallQuadr, 'color','blue');
hold on;
plot((1-SpecLinear),RecallLinear, 'color','red');
hold off;

%% Diferencias significativas entre modelos
ACCMeanLinear = mean(ACCLinear,2); % para cada modelo calculamos la ACC media para las tres clases de flor
ACCMeanQuadr = mean(ACCQuadr,2);

muestras = [ACCMeanLinear,ACCMeanQuadr];
etiquetas = ['linear';'quadra'];
[P] = testEstadistico(muestras,etiquetas,0.05);