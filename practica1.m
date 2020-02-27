%% limpiar  
clear all
%cambiar la semilla 
rng('shuffle')
%% cargamos la base de datos de Iris
run loadIris
inputs = [VarName1, VarName2, VarName3, VarName4];
outputs = Irissetosa;
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
plot((1-SpecQuadr),RecallQuadr, 'color','blue');
hold on;
plot((1-SpecLinear),RecallLinear, 'color','red');
hold off;

%% Diferencias significativas entre modelos
ACCMeanLinear = mean(ACCLinear,2); % para cada modelo calculamos la ACC media para las tres clases de flor
ACCMeanQuadr = mean(ACCQuadr,2);

muestras = [ACCMeanLinear,ACCMeanQuadr];
etiquetas = ['linear';'quadra'];
[P] = testEstadistico(muestras,etiquetas,0.05)

%% cargamos la base de datos de cancer
run loadCancer
inputs = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9];
outputs = ClaseCancer;

%% eliminar valores nulos (se podría rellenar con valores medios)
dataset = [inputs,outputs]
%filas con valores nulos
rows = any(isnan(inputs),2);
%eliminación de valores nulos
dataset(rows,:) = [];
inputs = dataset(:,1:9)
outputs = dataset(:,10)
%% normalización
inputs = (inputs - mean(mean(inputs)))/std(std(inputs));