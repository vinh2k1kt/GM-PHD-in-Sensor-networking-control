function [meas, obs_num] = gen_meas(model, hasMeasNoise, simu_time, ground_truth)

    duration = size(ground_truth,2);

    obs_num = zeros(1, simu_time);

    meas = cell(duration, 1);

    noise = zeros(model.z_dim,duration);

    if (hasMeasNoise)
        noise = mvnrnd(zeros(2,1), model.R, duration)';
    end

    for i = 1 : size(ground_truth,2)
       if(rand(1) <= model.P_D)
            meas{i} = model.H * ground_truth(:,i) + noise(:,i);
            obs_num(i) = 1;
       end
    end

end