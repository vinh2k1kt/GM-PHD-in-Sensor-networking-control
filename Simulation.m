% ---------------------Sensor Network Control--------------------
% -----Using Gaussian Mixture Probability Hypothesis Density-----
% -----------------------------------------------------------------
% -----------------------------------------------------------------

clc, clear, close all

%% Simulation Setting

Duration = 50;
sur_area = [0 0; 1000 1000]; %Survilance area [x_min y_min; x_max y_max] 
sensor_spacing = [50; 50];   %Space between each sensor [x_space; y_space]

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

model = Gen_model;
model.range_c = sur_area;
model.pdf_c = 1/prod(model.range_c(2,:)-model.range_c(1,:));

%% Generate Ground Truth

gt_1 = Gen_ground_truth('Linear',obj_1,Duration,model);
gt_2 = Gen_ground_truth('Linear', obj_2, Duration,model);

gt_1 = Hyper_box(sur_area, gt_1);
gt_2 = Hyper_box(sur_area, gt_2);

%% Plot Scenario

figure(1);
hold on;
    
for row = 1 : num_of_sensor(2)
    for col = 1: num_of_sensor(1)
        sensor = sensor_network(row,col);
        plot(sensor.x, sensor.y, 'b.', 'LineWidth', 1.5, 'MarkerSize', 5);
    end
end

gt1_plot = plot(gt_1(1,:), gt_1(2,:), '--rp', 'LineWidth', 1.5, 'MarkerSize', 5);
gt2_plot = plot(gt_2(1,:), gt_2(2,:), '--yh', 'LineWidth', 1.5, 'MarkerSize', 5);
