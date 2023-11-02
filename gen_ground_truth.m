function [ground_truth] = Gen_ground_truth(type,starting_point,duration,model)

    ground_truth = zeros(model.x_dim, duration);
    
    switch type
        case 'Linear'
            ground_truth(:,1) = starting_point;
            for i = 2 : duration
                ground_truth(:,i) = model.F * ground_truth(:,i-1);
            end
        otherwise
            disp("Wrong input argument");
    end
end