function [ECM] = calcularECM(cvTraining, cvTest, inputs, outputs, mdls, flag_training )
%% predecimos con el modelo
for i = 1:size(cvTraining,1)
    %en caso de entrenamiento se predice con los datos de training
    if (flag_training)
        idx = cvTraining(i,:);
    else
        idx = cvTest(i,:);    
    end
    pred = predict(mdls{i}, inputs(idx,:));
    ECM(i) = immse(pred,outputs(idx));
end

end