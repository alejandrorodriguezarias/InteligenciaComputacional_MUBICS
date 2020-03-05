function [inputs_norm] = normalize(inputs)

for i = 1:size(inputs,2)
    inputs_norm(:,i) = (inputs(:,i) - mean(inputs(:,i)))/std(inputs(:,i));
end
