% ---------------------Sensor Network Control--------------------
% -----Using Gaussian Mixture Probability Hypothesis Density-----
% -----------------------------------------------------------------
% -----------------------------------------------------------------

clc, clear, close all

%% Simulation Setting

duration = 50;
sur_area = [0 0; 1000 1000]; %Survilance area [x_min y_min; x_max y_max] 
sensor_spacing = [50; 50];   %Space between each sensor [x_space; y_space]

hasMeasNoise = true;
hasClutter = true;

%% Object Setting (obj_k = [x;y;vx;vy])

obj_1 = [0;0;10;10];
obj_2 = [500;1000;10;-10];

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

%% Generate Clutter

clutter_num = zeros(1, duration);
clutter = cell(duration, 1);

if (hasClutter)

    for i = 1 : duration
        clutter_num(i) = poissrnd(model.lambda_c);
        clutter{i} = [unifrnd(model.range_c(1,1), model.range_c(2,1), 1, clutter_num(i))
                       unifrnd(model.range_c(1,2), model.range_c(2,2), 1, clutter_num(i))];

        z_1{i} = [z_1{i} clutter{i}];
        z_2{i} = [z_2{i} clutter{i}];
    end

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

gt1_plot = plot(gt_1(1,:), gt_1(2,:), '-rp', 'LineWidth', 1.5, 'MarkerSize', 2);
gt2_plot = plot(gt_2(1,:), gt_2(2,:), '-yp', 'LineWidth', 1.5, 'MarkerSize', 2);

for i = 1 : duration
    %z1_plot = plot(z_1{i}(1,:), z_1{i}(2,:), '.k');
    %z2_plot = plot(z_2{i}(1,:), z_2{i}(2,:), '.k');
end
