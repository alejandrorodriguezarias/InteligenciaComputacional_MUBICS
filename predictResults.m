function [Recall,Spec,Precision,NPV,ACC,F1Score, prediction] = predictResults(cv, inputs, outputs, mdls, flag_training )
%% predecimos con el modelo
for i = 1:cv.NumTestSets
    if (flag_training)
        idx = cv.training(i);
    else
        idx = cv.test(i);    
    end
    prediction{i} = predict(mdls{i}, inputs(idx,:));
    [CM, orderCM] = confusionmat(outputs(idx,:), prediction{i});
    if size(CM,1) > 2
        for j = 1:size(CM,1)
            [Recall(i,j),Spec(i,j),Precision(i,j),NPV(i,j),ACC(i,j),F1Score(i,j)] = performance_indexes(CM,j);
        end
    else
        [Recall(i),Spec(i),Precision(i),NPV(i),ACC(i),F1Score(i)] = performance_indexes(CM,2); 
    end    
end

end