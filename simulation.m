% ---------------------Sensor Network Control--------------------
% -----Using Gaussian Mixture Probability Hypothesis Density-----
% -----------------------------------------------------------------
% -----------------------------------------------------------------

clc, clear, close all

%% Simulation Setting

duration = 100;
sur_area = [0 0; 1000 1000]; %Survilance area [x_min y_min; x_max y_max] 
sensor_spacing = [100; 100];   %Space between each sensor [x_space; y_space]

hasMeasNoise = true;
hasClutter = true;
hasBirthObj = false;

doPlotOSPA = false;
doPlotEstimation = true;
doPlotSensorNetwork = false;
doPlotSensorNetworkProcess =true;
doPlotVoidProb = false;

%% Object Setting (obj_k = [x;y;vx;vy])

obj_1 = [0;250;2;0];
obj_2 = [0;500;2;0];

birth_obj_1 = [500;0;0;2];
birth_time_1 = 30;
b_duration_1 = duration - min(duration, birth_time_1);

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
model.pdf_c = 1/prod(model.range_c(2,:) - model.range_c(1,:));

%% Generate Ground Truth

gt_1 = gen_ground_truth('Linear',obj_1,duration,model);
gt_2 = gen_ground_truth('Linear', obj_2, duration,model);

gt_1 = hyper_box(sur_area, gt_1);
gt_2 = hyper_box(sur_area, gt_2);

birth_gt_1 = gen_ground_truth('Linear', birth_obj_1, b_duration_1, model);

birth_gt_1 = hyper_box(sur_area, birth_gt_1);

b_duration_1 = size(birth_gt_1, 2);

%% Generate Measurement

[z_1, o_num_1] = gen_meas(model, hasMeasNoise, duration, gt_1);
[z_2, o_num_2] = gen_meas(model, hasMeasNoise, duration, gt_2);

[b_1, b_num_1] = gen_meas(model, hasMeasNoise, b_duration_1, birth_gt_1);

% Padding 

z_1 = [z_1' cell(duration - size(z_1, 1), 1)']';
z_2 = [z_2' cell(duration - size(z_2, 1), 1)']';

b_1 = [cell(birth_time_1, 1)', b_1', cell(duration - b_duration_1 - birth_time_1, 1)']';

%Merging

z = cell(duration, 1);
for i = 1 : duration

    if (hasBirthObj)
        z{i} = [z_1{i} z_2{i} b_1{i}];
    else
        z{i} = [z_1{i} z_2{i}];
    end
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

sensor_trajectory = repmat([1;1],1,duration);

%% Pruning & Merging Parameter Setting

elim_threshold = 1e-5;        % pruning threshold
merge_threshold = 4;          % merging threshold
L_max = 100;                  % limit on number of Gaussian components

%% Recursion

for k = 2:duration

    %% Predict

    [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
    w_predict = model.P_S * w_update{k-1};

    w_predict = cat(1, model.w_birth, w_predict);
    m_predict = cat(2, model.m_birth, m_predict);
    P_predict = cat(3, model.P_birth, P_predict);

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

    [sensor_trajectory(:,k), min_void_probability, void_matrix] = void_prob_rec(sensor_trajectory(:,k-1), sensor_network, est_w{k}, ...
                                 est_state{k}, est_cov{k}, sur_area);
    
    void_prob(k) = min_void_probability;
    void_prob_matrix{k} = void_matrix;

    if (doPlotSensorNetworkProcess)

        figure(3);
        hold on;

        for r = 1 : num_of_sensor(2)
            for c = 1 : num_of_sensor(1)
                sensor_plot = plot(sensor_network(r,c).x, sensor_network(r,c).y, '--x' ...
                , 'LineWidth', 1.5, 'MarkerSize', 5 ...
                ,'Color', 'black'); 
            end
        end
        
        current_pos = sensor_trajectory(:,k);
        current_sensor_plot = plot(sensor_network(current_pos(1), current_pos(2)).x, ...
                                   sensor_network(current_pos(1), current_pos(2)).y, ...
                                   'LineWidth', 1.5, 'MarkerSize', 10, ...
                                   'MarkerFaceColor', [1 0.4784 0.4784], ...
                                   'MarkerEdgeColor', 'red', ...
                                   'Marker', 'diamond');

        if (k <= size(gt_1, 2))
            gt1_plot = plot(gt_1(1,k), gt_1(2,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color',[0.9290 0.6940 0.1250]);
        end
        
        if (k <= size(gt_2, 2))
            gt2_plot = plot(gt_2(1,k), gt_2(2,k), '--o', 'LineWidth', 1.5, 'MarkerSize', 5 ...
            ,'Color', [1 0 0]);
        end

        if (hasBirthObj && k > birth_time_1 && k < birth_time_1 + b_duration_1)
            b1_plot = plot(birth_gt_1(1,k-birth_time_1), birth_gt_1(2,k-birth_time_1), ...
             '--o', 'LineWidth', 1.5, 'MarkerSize', 5, ...
             'Color',[0.1490 0.9882 0.7216]);
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

%         if (hasBirthObj)
%             legend([gt1_plot, gt2_plot, b1_plot,est_plot, sensor_plot], ...
%             'Ground truth 1', 'Ground truth 2', 'Birth 1', 'Estimated State', 'Sensor', ...
%             'Location', 'northeastoutside');
%         else
%             legend([gt1_plot, gt2_plot, est_plot, sensor_plot], ...
%             'Ground truth 1', 'Ground truth 2', 'Estimated State', 'Sensor', ...
%             'Location', 'northeastoutside');
%         end

        pause(0.01);
    end
end

%% Visualize

if (doPlotEstimation)
    figure(1);
    
    hold on; 
    
    for row = 1 : num_of_sensor(2)
        for col = 1: num_of_sensor(1)
            
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

    for r = 1 : num_of_sensor(2)
        for c = 1 : num_of_sensor(1)
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