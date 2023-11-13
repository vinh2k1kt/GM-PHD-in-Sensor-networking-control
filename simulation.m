% ---------------------Sensor Network Control--------------------
% -----Using Gaussian Mixture Probability Hypothesis Density-----
% -----------------------------------------------------------------
% -----------------------------------------------------------------

clc, clear, close all

%% Simulation Setting

duration = 100;
sur_area = [0 0; 1000 1000]; %Survilance area [x_min y_min; x_max y_max] 
sensor_spacing = [50; 50];   %Space between each sensor [x_space; y_space]

hasClutter = true;
hasBirthObj = false;

doPlotOSPA = false;
doPlotEstimation = false;
doPlotSensorNetwork = false;
doPlotSensorNetworkProcess = true;
doPlotVoidProb = false;

%% Generate Sensor Coordination

sensor_num = [sur_area(2,1)/sensor_spacing(1) + 1; sur_area(2,2)/sensor_spacing(2) + 1];

sensor.pos = 0;

sensor_network = repmat(sensor,sensor_num(2), sensor_num(1));

for row = 1 : sensor_num(2)
    for col = 1: sensor_num(1)
        sensor_network(row,col).pos = ...
            [(col - 1) * sensor_spacing(1);(row -1) * sensor_spacing(2)];
    end
end

%% Generate Model

model = gen_model;
model.range_c = sur_area;
model.pdf_c = 1/prod(model.range_c(2,:) - model.range_c(1,:));


%% Object Setting (obj_k = [x;y;vx;vy])

t_die = 19;

obj_1 = [800; 600; -.3; -1.8];
obj_2 = [650; 500; -.4; 1.1];
obj_3 = [600; 720; .75; -1.5];
obj_4 = [700; 700; 1.7; .2];
obj_5 = [750; 800; 1.6; -1.2];

obj = cat(2,obj_1, obj_2);

tar_status = true(size(obj,2), duration);
tar_status(size(obj,2),t_die+1:end) = false;

%% Generate Ground Truth

gt = zeros(size(obj,1), size(obj,2), duration);

[gt(:,1,:), tar_status(1,:)] = gen_ground_truth('Linear', obj_1, duration,model, sur_area);
[gt(:,2,:), tar_status(2,:)] = gen_ground_truth('Linear', obj_2, duration,model, sur_area);
[gt(:,3,:), tar_status(3,:)] = gen_ground_truth('Linear', obj_3, duration,model, sur_area);
[gt(:,4,:), tar_status(4,:)] = gen_ground_truth('Linear', obj_4, duration,model, sur_area);
[gt(:,5,:), tar_status(5,:)] = gen_ground_truth('Linear', obj_5, duration,model, sur_area);
    
tar_status(5,min(t_die+1,sum(tar_status(5,:), 'all')+1):end) = false;

%% Generate Clutter

clutter_num = zeros(1, duration);
clutter = cell(duration, 1);

if (hasClutter)
    for i = 1 : duration
        clutter_num(i) = poissrnd(model.lambda_c);
        clutter{i} = [unifrnd(model.range_c(1,1), model.range_c(2,1), 1, clutter_num(i))
                      unifrnd(model.range_c(1,2), model.range_c(2,2), 1, clutter_num(i))];
    end
end

%% Prior Initialze

w_update = cell(duration, 1);
m_update = cell(duration, 1);
P_update = cell(duration, 1);
P_D = cell(duration, 1);

w_update{1} = 0.5;
m_update{1}(:, 1) = [1000; 1000; 10; 10];
P_update{1}(:, :, 1) = diag([sur_area(2,1) sur_area(2,2) 100 100]).^2;

L_update = 1;
est_state = cell(duration, 1);
est_cov = cell(duration, 1);
est_w = cell(duration, 1);

num_objects = zeros(duration, 1);
void_prob = zeros(duration, 1);
void_prob_matrix = cell(duration, 1);

sensor_traj = repmat([sensor_num(2);sensor_num(1)],1,duration);
% sensor_traj = repmat([15;15],1,duration);

%% Initial Prediction

[m_predict, P_predict] = predict_KF(model, m_update{1}, P_update{1});
w_predict = model.P_S * w_update{1};

w_predict = cat(1, model.w_birth, w_predict);
m_predict = cat(2, model.m_birth, m_predict);
P_predict = cat(3, model.P_birth, P_predict);

%% Pruning & Merging Parameter Setting

elim_threshold = 1e-3;        % pruning threshold
merge_threshold = 4;          % merging threshold
L_max = 100;                  % limit on number of Gaussian components

%% Recursion

for k = 1:duration
    
    sensor_pos = sensor_network(sensor_traj(1,k), sensor_traj(2,k)).pos;
    [z, P_D{k}] = get_meas(sensor_pos, gt(:,:,k),model,clutter{k});
    
    %% Update
    n = size(z,2);       %number of measurement

    % Miss Detection Hypothesis

    diff = repmat(sensor_pos,1,size(m_predict(1:2,:),2)) - m_predict(1:2,:);
    P_D_predict = zeros(1,size(m_predict(1:2,:),2));
    for idx = 1 : size(P_D_predict,2)
        P_D_predict(:,idx) = exp(-.5*diff(:,idx)'*model.P_D_cov_inv*diff(:,idx));
    end

    w_update{k} = model.P_MD*w_predict;
    m_update{k} = m_predict;
    P_update{k} = P_predict;

    % Detected Hypothesis
    [likelihood_tmp] = cal_likelihood(z,model,m_predict,P_predict);

    if n ~= 0
        [m_temp, P_temp] = update_KF(z,model,m_predict,P_predict);
        for i = 1:n
            
            % Calculate P_D
            diff = repmat(sensor_pos,1,size(m_temp(1:2,:,i),2)) - m_temp(1:2,:,i);
            P_D_temp = zeros(1,size(m_temp(1:2,:,i),2));
            for idx = 1 : size(P_D_temp,2)
                P_D_temp(:,idx) = exp(-.5*diff(:,idx)'*model.P_D_cov_inv*diff(:,idx));
            end
    
            % Calculate detection weight of each probable object detect
            w_temp = P_D_temp * w_predict .* likelihood_tmp(:,i);
            
            if (hasClutter)
                w_temp = w_temp ./ (model.lambda_c*model.pdf_c + sum(w_temp));
            else
                w_temp = w_temp ./ (sum(w_temp));
            end

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
    w_copy = w_update{k};
    indices = [];

    for i = 1:num_objects(k)
        [~, maxIndex] = max(w_copy);
        indices(i) = maxIndex;
        w_copy(maxIndex) = -inf;
    end

    for i = 1:size(indices,2)
        est_cov{k} = [cat(3, est_cov{k}, P_update{k}(:,:,indices(i)))];
        est_state{k} = [est_state{k} m_update{k}(:,indices(i))];
        est_w{k} = [est_w{k} w_update{k}(indices(i),:)];
    end

    %% Predict

    [m_predict, P_predict] = predict_KF(model, m_update{k}, P_update{k});
    w_predict = model.P_S * w_update{k};

    w_predict = cat(1, model.w_birth, w_predict);
    m_predict = cat(2, model.m_birth, m_predict);
    P_predict = cat(3, model.P_birth, P_predict);

    %% Calculate next optimize sensor position


    [sensor_traj(:,k+1), min_void_probability, void_matrix] = void_prob_rec(sensor_traj(:,k),...
     sensor_network, w_predict,m_predict, P_predict, sur_area, sensor_spacing);
    
    void_prob(k) = min_void_probability;
    void_prob_matrix{k} = void_matrix;

    if (doPlotSensorNetworkProcess)

        figure(3);
        hold on;

%         for r = 1 : sensor_num(2)
%             for c = 1 : sensor_num(1)
%                 sensor_plot = plot(sensor_network(r,c).pos(1,:), sensor_network(r,c).pos(2,:) ...
%                 , '--x' ...
%                 , 'LineWidth', 1.5, 'MarkerSize', 5 ...
%                 ,'Color', 'black'); 
%             end
%         end
%         
        current_pos = sensor_network(sensor_traj(1,k), sensor_traj(2,k)).pos;
        current_sensor_plot = plot(current_pos(1,:), current_pos(2,:), ...
                                   'LineWidth', 1.5, 'MarkerSize', 10, ...
                                   'MarkerFaceColor', [1 0.4784 0.4784], ...
                                   'MarkerEdgeColor', 'red', ...
                                   'Marker', 'diamond');

        if (k < sum(tar_status(1,:)))
            gt1_plot = plot(gt(1,1,k), gt(2,1,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color',[0.9290 0.6940 0.1250]);
        elseif (k == sum(tar_status(1,:)))
            gt1_plot = plot(gt(1,1,k), gt(2,1,k), '--^', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color',[0.9290 0.6940 0.1250]);
        end
        
        if (k < sum(tar_status(2,:)))
            gt2_plot = plot(gt(1,2,k), gt(2,2,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [1 0 0]);
        elseif (k == sum(tar_status(2,:)))
            gt2_plot = plot(gt(1,2,k), gt(2,2,k), '--^', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [1 0 0]);
        end

        if (k < sum(tar_status(3,:)))
            gt3_plot = plot(gt(1,3,k), gt(2,3,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [1.0000 0.4784 0.4784]);
        elseif (k == sum(tar_status(3,:)))
            gt3_plot = plot(gt(1,3,k), gt(2,3,k), '--^', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [1.0000 0.4784 0.4784]);
        end

        if (k < sum(tar_status(4,:)))
            gt4_plot = plot(gt(1,4,k), gt(2,4,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [0 1 0]);
        elseif (k == sum(tar_status(4,:)))
            gt4_plot = plot(gt(1,4,k), gt(2,4,k), '--^', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [0 1 0]);
        end

        if (k < sum(tar_status(5,:)))
            gt5_plot = plot(gt(1,5,k), gt(2,5,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [0.9137 0.0471 0.9608]);
        elseif (k == sum(tar_status(5,:)))
            gt5_plot = plot(gt(1,5,k), gt(2,5,k), '--^', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [0.9137 0.0471 0.9608]);
        end
        
        for num = 1 : num_objects(k)
            est_plot = plot(est_state{k}(1, num), est_state{k}(2, num), 'o' ...
                    , 'MarkerSize', 5, 'MarkerFaceColor', 'blue');
        end

        xlabel('x axis', 'FontSize', 12, 'FontWeight','bold');
        ylabel('y axis', 'FontSize', 12, 'FontWeight','bold');
        xlim([sur_area(1,1),sur_area(2,1)]);
        ylim([sur_area(1,2), sur_area(2,2)]);
        title('Estimation', ...
            'FontSize', 14, ...
            'FontWeight','bold');

        pause(0.01);
    end
end

%% Visualize

if (doPlotEstimation)
    figure(1);
    
    hold on; 
    
    for row = 1 : sensor_num(2)
        for col = 1: sensor_num(1)
            
        end
    end
    
    
    gt1_plot = plot(gt_1(1,:), gt_1(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color',[0.9290 0.6940 0.1250]);
    gt2_plot = plot(gt_2(1,:), gt_2(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color', [1 0 0]);
    
    if (hasBirthObj)
        b1_plot = plot(birth_gt_1(1,:), birth_gt_1(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color',[0.1490 0.9882 0.7216]);
    end
    
    for i = 1 : duration
        z_plot = plot(z{i}(1,:), z{i}(2,:), '.b');
    end
    
    xlabel('x axis', 'FontSize', 12, 'FontWeight','bold');
    ylabel('y axis', 'FontSize', 12, 'FontWeight','bold');
    xlim([sur_area(1,1),sur_area(2,1)]);
    ylim([sur_area(1,2), sur_area(2,2)]);
    title('Sensor POV', ...
        'FontSize', 14, ...
        'FontWeight', 'bold');
    
    if (hasBirthObj)
        legend([gt1_plot, gt2_plot, b1_plot, z_plot], ...
        'Ground truth 1', 'Ground truth 2', 'Birth 1', 'Measurement', ...
        'Location', 'northeastoutside');
    else
        legend([gt1_plot, gt2_plot, z_plot], ...
        'Ground truth 1', 'Ground truth 2', 'Measurement', ...
        'Location', 'northeastoutside');
    end
    
    figure(2);
    hold on;
    
    gt1_plot = plot(gt_1(1,:), gt_1(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color',[0.9290 0.6940 0.1250]);
    gt2_plot = plot(gt_2(1,:), gt_2(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color', [1 0 0]);
    
    if (hasBirthObj)
        b1_plot = plot(birth_gt_1(1,:), birth_gt_1(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color',[0.1490 0.9882 0.7216]);
    end
    
    for t = 2:duration
        for k = 1:num_objects(t)
            est_plot = plot(est_state{t}(1, k), est_state{t}(2, k), 'o' ...
                , 'MarkerSize', 5, 'MarkerFaceColor', 'blue');
        end
    end
    
    xlabel('x axis', 'FontSize', 12, 'FontWeight','bold');
    ylabel('y axis', 'FontSize', 12, 'FontWeight','bold');
    xlim([sur_area(1,1),sur_area(2,1)]);
    ylim([sur_area(1,2), sur_area(2,2)]);
    title('Estimation', ...
        'FontSize', 14, ...
        'FontWeight','bold');
    
    if (hasBirthObj)
        legend([gt1_plot, gt2_plot, b1_plot,est_plot], ...
        'Ground truth 1', 'Ground truth 2', 'Birth 1', 'Estimated State', ...
        'Location', 'northeastoutside');
    else
        legend([gt1_plot, gt2_plot, est_plot], ...
        'Ground truth 1', 'Ground truth 2', 'Estimated State', ...
        'Location', 'northeastoutside');
    end

end

if (doPlotSensorNetwork)
    
    figure(3);
    hold on;

    for r = 1 : sensor_num(2)
        for c = 1 : sensor_num(1)
            sensor_plot = plot(sensor_network(r,c).x, sensor_network(r,c).y, '--x' ...
            , 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', 'black');
        end
    end
    
        gt1_plot = plot(gt_1(1,:), gt_1(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color',[0.9290 0.6940 0.1250]);
        gt2_plot = plot(gt_2(1,:), gt_2(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color', [1 0 0]);
    
    if (hasBirthObj)
        b1_plot = plot(birth_gt_1(1,:), birth_gt_1(2,:), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
        ,'Color',[0.1490 0.9882 0.7216]);
    end
    
    for t = 2:duration
        for k = 1:num_objects(t)
            est_plot = plot(est_state{t}(1, k), est_state{t}(2, k), 'o' ...
                , 'MarkerSize', 5, 'MarkerFaceColor', 'blue');
        end
    end
    
    xlabel('x axis', 'FontSize', 12, 'FontWeight','bold');
    ylabel('y axis', 'FontSize', 12, 'FontWeight','bold');
    xlim([sur_area(1,1),sur_area(2,1)]);
    ylim([sur_area(1,2), sur_area(2,2)]);
    title('Estimation', ...
        'FontSize', 14, ...
        'FontWeight','bold');
    
    if (hasBirthObj)
        legend([gt1_plot, gt2_plot, b1_plot,est_plot, sensor_plot], ...
        'Ground truth 1', 'Ground truth 2', 'Birth 1', 'Estimated State', 'Sensor', ...
        'Location', 'northeastoutside');
    else
        legend([gt1_plot, gt2_plot, est_plot, sensor_plot], ...
        'Ground truth 1', 'Ground truth 2', 'Estimated State', 'Sensor', ...
        'Location', 'northeastoutside');
    end
end

%% Evaluaion

if (doPlotOSPA)

    ospa = zeros(1, duration);
    ospa_cutoff = 100;
    ospa_order = 2;
    
    for t = 2:duration
    
        if (~isempty(est_state{t})) 
            est_mat = est_state{t}(1:2,:);
        else
            est_mat = [];
        end
        
        if ~(t > size(gt_1, 2))
            gt1_mat = gt_1(1:2,t);
        else
            gt1_mat = [];
        end
    
        if ~(t > size(gt_2, 2))
            gt2_mat = gt_2(1:2,t);
        else
            gt2_mat = [];
        end
    
        if (t <= birth_time_1 || t > (birth_time_1 + b_duration_1))
            b1_mat = [];
        else
            b1_mat = birth_gt_1(1:2, t - birth_time_1);
        end
        
        if (hasBirthObj)
            ospa(t) = ospa_dist([gt1_mat, gt2_mat, b1_mat], est_mat, ospa_cutoff, ospa_order);
        else
            ospa(t) = ospa_dist([gt1_mat, gt2_mat], est_mat, ospa_cutoff, ospa_order);
        end
    end
    
    figure (4);
    hold on;
        
    plot(2:duration, ospa(2:end), 'LineWidth', 1, 'Color', [0 1 1]);
    
    xlabel('Time step');
    ylabel('Distance (in m)');
    title('OSPA Evaluation', 'FontWeight', 'bold');

end

if (doPlotVoidProb)

    figure(5);
    plot(void_prob, 'LineWidth', 1, 'Color', 'red');

    xlabel('Time step');
    ylabel('Void Probability');
    title('Void Probability', 'FontWeight', 'bold');
end