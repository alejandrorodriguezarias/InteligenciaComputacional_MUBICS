function [Recall,Spec,Precision,NPV,ACC,F1Score, prediction] = predictResultsLOO(cv, inputs, outputs, mdls)
%% predecimos con el modelo para valores de test
for i = 1:cv.NumTestSets
    idx = cv.test(i);    
    prediction(i) = predict(mdls{i}, inputs(idx,:));
    realValue(i) = outputs(idx,:);
end
for i = 1:10
    %dividimos los 150 valores en grupos de 15
    [CM, orderCM] = confusionmat(realValue((i-1)*15+1:i*15), prediction((i-1)*15+1:i*15));
    if size(CM,1) > 2
        for j = 1:size(CM,1)
            [Recall(i,j),Spec(i,j),Precision(i,j),NPV(i,j),ACC(i,j),F1Score(i,j)] = performance_indexes(CM,j);
        end
    else
        [Recall(i),Spec(i),Precision(i),NPV(i),ACC(i),F1Score(i)] = performance_indexes(CM,2); 
    end  

end