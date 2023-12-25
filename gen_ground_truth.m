function [ground_truth, tar_status] = gen_ground_truth(type,starting_point,duration,model, sur_area)

    ground_truth = zeros(model.x_dim, duration);
    tar_status = true(1, duration);
    switch type
        case 'Linear'
            ground_truth(:,1) = starting_point;
            for i = 2 : duration
                ground_truth(:,i) = model.F * ground_truth(:,i-1);
            end
            ground_truth = hyper_box(sur_area, ground_truth);
            tar_status(1,size(ground_truth,2)+1:end) = false;
            
            ground_truth = [ground_truth, zeros(model.x_dim,duration - size(ground_truth,2))];
            ground_truth = reshape(ground_truth,[model.x_dim,1,duration]);
        otherwise
            disp("Wrong input argument");
    end
end