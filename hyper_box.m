function [target_in_box] = hyper_box(box, target)
    
    coordinates = target(1:2,:); 

    lower_bound = repmat(box(1,:)',1,size(target,2));
    upper_bound = repmat(box(2,:)',1,size(target,2));

    isInHyperBox = all (coordinates <= upper_bound & coordinates >= lower_bound);
    
    target_in_box = target(:, isInHyperBox);
end