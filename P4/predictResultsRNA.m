function [Recall,Spec,Precision,NPV,ACC,F1Score, prediction] = predictResultsRNA(cv, inputs, outputs, mdls, flag_training )
%% predecimos con el modelo
for i = 1:cv.NumTestSets
    %en caso de entrenamiento se predice con los datos de training
    if (flag_training)
        idx = cv.training(i);
    else
        idx = cv.test(i);    
    end
    prediction{i} = round(mdls{i}(inputs(:,idx)));
    [CM, orderCM] = confusionmat(outputs(:,idx), prediction{i});
    %+ de dos clases
    if size(CM,1) > 2
        for j = 1:size(CM,1)
            [Recall(i,j),Spec(i,j),Precision(i,j),NPV(i,j),ACC(i,j),F1Score(i,j)] = performance_indexes(CM,j);
        end
    %2 clases
    else
        [Recall(i),Spec(i),Precision(i),NPV(i),ACC(i),F1Score(i)] = performance_indexes(CM,2); 
    end    
end
plotconfusion(outputs(:,idx),prediction{i})
%     errors = gsubtract(outputs,resultado);
%     performance = perform(net,outputs,resultado);
end