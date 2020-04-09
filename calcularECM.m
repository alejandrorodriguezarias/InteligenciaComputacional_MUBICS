function [ECM] = calcularECM(cvTraining, cvTest, inputs, outputs, mdls, flag_training )
%% predecimos con el modelo
for i = 1:size(cvTraining,1)
    %en caso de entrenamiento se predice con los datos de training
    if (flag_training)
        idx = cvTraining(i,:);
    else
        idx = cvTest(i,:);    
    end
    ECM{i} = sum(predict(mdls{i}, inputs(idx,:)) - outputs(idx))/size(outputs(idx),1);
    
end

end