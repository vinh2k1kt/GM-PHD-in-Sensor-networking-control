% ---------------------Sensor Network Control--------------------
% -----Using Gaussian Mixture Probability Hypothesis Density-----
% -----------------------------------------------------------------
% -----------------------------------------------------------------

clc, clear, close all

%% Simulation Setting

duration = 100;
sur_area = [0 0; 1000 1000]; %Survilance area [x_min y_min; x_max y_max] 
sensor_spacing = [50; 50];   %Space between each sensor [x_space; y_space]

hasMeasNoise = true;
hasClutter = true;

%% Object Setting (obj_k = [x;y;vx;vy])

obj_1 = [0;0;1;1];
obj_2 = [0;1000;1;-1];

%% Pruning & Merging Parameter Setting

elim_threshold = 1e-5;        % pruning threshold
merge_threshold = 4;          % merging threshold
L_max = 100;                  % limit on number of Gaussian components

%% Generate Sensor Coordination

num_of_sensor = [sur_area(2,1)/sensor_spacing(1) + 1; sur_area(2,2)/sensor_spacing(2) + 1];

cordinates.x = 0;
cordinates.y = 0;

sensor_network = repmat(cordinates,num_of_sensor(2), num_of_sensor(1));

for row = 1 : num_of_sensor(2)
    for col = 1: num_of_sensor(1)
        sensor_network(row,col).x = (col - 1) * sensor_spacing(1);
        sensor_network(row,col).y = (row -1) * sensor_spacing(2);
    end
end

%% Generate Model

model = gen_model;
model.range_c = sur_area;
model.pdf_c = 1/prod(model.range_c(2,:)-model.range_c(1,:));

%% Generate Ground Truth

gt_1 = gen_ground_truth('Linear',obj_1,duration,model);
gt_2 = gen_ground_truth('Linear', obj_2, duration,model);

gt_1 = hyper_box(sur_area, gt_1);
gt_2 = hyper_box(sur_area, gt_2);

%% Generate Measurement

[z_1, o_num_1] = gen_meas(model, hasMeasNoise, duration, gt_1);
[z_2, o_num_2] = gen_meas(model, hasMeasNoise, duration, gt_2);

% Padding 
z_1 = [z_1' cell(duration - size(z_1, 1), 1)']';
z_2 = [z_2' cell(duration - size(z_2, 1), 1)']';

%Merging

z = cell(duration, 1);
for i = 1 : duration
    z{i} = [z_1{i} z_2{i}];
end

%% Generate Clutter

clutter_num = zeros(1, duration);
clutter = cell(duration, 1);

if (hasClutter)

    for i = 1 : duration
        clutter_num(i) = poissrnd(model.lambda_c);
        clutter{i} = [unifrnd(model.range_c(1,1), model.range_c(2,1), 1, clutter_num(i))
                       unifrnd(model.range_c(1,2), model.range_c(2,2), 1, clutter_num(i))];

        z{i} = [z{i} clutter{i}];
    end

end

%% Prior Initialze

w_update{1} = [0.5; 0.5];

m_update{1}(:, 1) = [100; 100; 10; 10];
P_update{1}(:, :, 1) = diag([100 100 100 100]).^2;

m_update{1}(:, 2) = [200; 100; 10; 10];
P_update{1}(:, :, 2) = diag([100 100 100 100]).^2;

L_update = 2;
est_state = cell(duration, 1);
num_objects = zeros(duration, 1);

%% Recursion

for k = 2:duration

    %% Predict

    [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
    w_predict = model.P_S * w_update{k-1};

    m_predict = cat(2, model.m_birth, m_predict);
    P_predict = cat(3, model.P_birth, P_predict);
    w_predict = cat(1, model.w_birth, w_predict);

    %% Update
    n = size(z{k},2);       %number of measurement

    % Miss Detection Hypothesis
    w_update{k} = model.P_MD*w_predict;
    m_update{k} = m_predict;
    P_update{k} = P_predict;

    % Detected Hypothesis
    [likelihood_tmp] = cal_likelihood(z{k},model,m_predict,P_predict);

    if n ~= 0
        [m_temp, P_temp] = update_KF(z{k},model,m_predict,P_predict);
        for i = 1:n

            % Calculate detection weight of each probable object detect
            w_temp = model.P_D * w_predict .* likelihood_tmp(:,i);
            w_temp = w_temp ./ (model.lambda_c*model.pdf_c + sum(w_temp));

            % Cat all of them to a vector of weight
            w_update{k} = cat(1,w_update{k},w_temp);

            % Update mean and covariance
            m_update{k} = cat(2,m_update{k},m_temp(:,:,i));
            P_update{k} = cat(3,P_update{k},P_temp);
        end
    end


    %---mixture management
    L_posterior= length(w_update{k});
    
    %% pruning, merging, caping
    [w_update{k},m_update{k},P_update{k}]= gaus_prune(w_update{k},m_update{k},P_update{k},elim_threshold);    
    L_prune= length(w_update{k});
    [w_update{k},m_update{k},P_update{k}]= gaus_merge(w_update{k},m_update{k},P_update{k},merge_threshold);   
    L_merge= length(w_update{k});
    [w_update{k},m_update{k},P_update{k}]= gaus_cap(w_update{k},m_update{k},P_update{k},L_max);               
    L_cap= length(w_update{k});
    
    L_update= L_cap;
    
    %% Estimate object state
    num_objects(k) = round(sum(w_update{k}));
    num_targets = num_objects(k);
    w_copy = w_update{k};
    indices = [];

    for i = 1:num_objects(k)
        [~, maxIndex] = max(w_copy);
        indices(i) = maxIndex;
        w_copy(maxIndex) = -inf;
    end

    for i = 1:size(indices,2)
        est_state{k} = [est_state{k} m_update{k}(:,i)];
    end

    %---display diagnostics
    disp([' time= ',num2str(k),...
         ' #gaus orig=',num2str(L_posterior),...
         ' #gaus elim=',num2str(L_prune), ...
         ' #gaus merg=',num2str(L_merge), ...
         ' #gaus cap=',num2str(L_cap), ...
         ' #measurement number=',num2str(n)]);
end

%% Plot Scenario

figure(1);
hold on;
    
for row = 1 : num_of_sensor(2)
    for col = 1: num_of_sensor(1)
        sensor = sensor_network(row,col);
        plot(sensor.x, sensor.y, '--kx', 'LineWidth', 1.5, 'MarkerSize', 5);
    end
end

gt1_plot = plot(gt_1(1,:), gt_1(2,:), '--rp', 'LineWidth', 1.5, 'MarkerSize', 2);
gt2_plot = plot(gt_2(1,:), gt_2(2,:), '--rp', 'LineWidth', 1.5, 'MarkerSize', 2);

for t = 2:duration
    for k = 1:num_objects(t)
        est_plot = plot(est_state{t}(1, k), est_state{t}(2, k), 'go');
    end
end

for i = 1 : duration
    z_plot = plot(z{i}(1,:), z{i}(2,:), '.b');
end
