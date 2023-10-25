function [ground_truth, orient] = Gen_ground_truth(Type,starting_point,Duration,model)
    ground_truth = zeros(model.x_dim, Duration);
    
    switch Type
        case 'Linear'
            ground_truth(:,1) = starting_point;
            for i = 2 : Duration
                ground_truth(:,i) = model.F * ground_truth(:,i-1);
            end
        otherwise
            disp("Wrong input argument");
    end
end