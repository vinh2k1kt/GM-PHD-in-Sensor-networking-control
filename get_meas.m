function [meas, P_D] = get_meas(sensor_pos, ground_truth, model, clutter)
    
    % p(x|m,P) = N(m,P) =
    % (2pi)^(-D/2)*det(P)^(-1/2) * exp(-1/2*(x-m)' * P^(-1) * (x-m))
    
    %calculate predicted measurement
    mu = model.H * ground_truth;

    diff = repmat(sensor_pos,1,size(mu,2)) - mu;
    P_D = zeros(1,size(mu,2));
    for i = 1 : size(mu,2)
        P_D(:,i) = exp(-.5*diff(:,i)'*model.P_D_cov_inv*diff(:,i));
    end
    
    t_dtd = rand(1,size(ground_truth,2)) < P_D;

    if any(t_dtd)
        meas(:,t_dtd)= mvnrnd(ground_truth(1:2,t_dtd)',model.R)';
        meas = [meas clutter];
    else
        meas = clutter;
    end
end