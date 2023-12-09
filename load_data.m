clc, clear, close all

load avg_simulation_data.mat;

figure(1);

hold on; 
sensor_traj_coor = zeros(2,duration);
for i = 1 :duration
    sensor_traj_coor(:,i) = sensor_network(sensor_traj(1,i), sensor_traj(2,i)).pos';
end

gt_1 = permute(gt(1:2,1,:),[1 3 2]);
gt_1 = gt_1(:,1:sum(tar_status(1,:)));

plot(gt_1(1,1), gt_1(2,1), 's', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333]) 
plot(gt_1(1,end), gt_1(2,end), 'v', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333])

gt1_plot = plot(gt_1(1,:), gt_1(2,:), '--', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333]);

gt_2 = permute(gt(1:2,2,:),[1 3 2]);
gt_2 = gt_2(:,1:sum(tar_status(2,:)));

plot(gt_2(1,1), gt_2(2,1), 's', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333]) 
plot(gt_2(1,end), gt_2(2,end), 'v', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333]) 

gt2_plot = plot(gt_2(1,:), gt_2(2,:), '--', 'LineWidth', 1.5 ...
,'Color', [0.3020 0.7451 0.9333]);

gt_3 = permute(gt(1:2,3,:),[1 3 2]);
gt_3 = gt_3(:,1:sum(tar_status(3,:)));

plot(gt_3(1,1), gt_3(2,1), 's', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333]) 
plot(gt_3(1,end), gt_3(2,end), 'v', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333])

gt3_plot = plot(gt_3(1,:), gt_3(2,:), '--', 'LineWidth', 1.5 ...
,'Color', [0.3020 0.7451 0.9333]);

gt_4 = permute(gt(1:2,4,:),[1 3 2]);
gt_4 = gt_4(:,1:sum(tar_status(4,:)));

plot(gt_4(1,1), gt_4(2,1), 's', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333]) 
plot(gt_4(1,end), gt_4(2,end), 'v', 'LineWidth', 1.5 ...
,'Color',[0.3020 0.7451 0.9333])

gt4_plot = plot(gt_4(1,:), gt_4(2,:), '--', 'LineWidth', 1.5 ...
,'Color', [0.3020 0.7451 0.9333]);

gt_5 = permute(gt(1:2,5,:),[1 3 2]);
gt_5 = gt_5(:,1:sum(tar_status(5,:)));

plot(gt_5(1,1), gt_5(2,1), 's', 'LineWidth', 1.5 ...
,'Color',[1 0 0]) 
plot(gt_5(1,end), gt_5(2,end), 'v', 'LineWidth', 1.5 ...
,'Color',[1 0 0])

gt5_plot = plot(gt_5(1,:), gt_5(2,:), '--', 'LineWidth', 1.5 ...
,'Color', [1 0 0]);

gt_6 = permute(gt(1:2,6,:),[1 3 2]);
gt_6 = gt_6(:,t_birth:t_birth - 1 + sum(tar_status(6,:)));

plot(gt_6(1,1), gt_6(2,1), 's', 'LineWidth', 1.5 ...
,'Color',[0 1 0]) 
plot(gt_6(1,end), gt_6(2,end), 'v', 'LineWidth', 1.5 ...
,'Color',[0 1 0])

gt6_plot = plot(gt_6(1,:), gt_6(2,:), '--', 'LineWidth', 1.5 ...
,'Color', [0 1 0]);

sensor_traj_plot = plot(sensor_traj_coor(1,:), sensor_traj_coor(2,:), ...
    'LineWidth', 1.5, ...
    'Color', [0.1490 0.9882 0.7216]);
plot(sensor_traj_coor(1,end), sensor_traj_coor(2,end), ...
    'LineWidth', 1.5, ...
    'Color', [0.1490 0.9882 0.7216], ...
    'Marker', 'diamond', ...
    'MarkerSize', 5);
xlabel('x coordiante (m)', 'FontSize', 12, 'FontWeight','bold');
ylabel('y coordiante (m)', 'FontSize', 12, 'FontWeight','bold');
xlim([sur_area(1,1),sur_area(2,1)]);
ylim([sur_area(1,2), sur_area(2,2)]);
legend(sensor_traj_plot, 'Sensor Trajectory', 'location', 'northeast');

hold off;

figure(6)

plot(2:duration, avg_ospa(2:duration), 'LineStyle','-','Color','red', 'LineWidth',1.5);

xlabel('Time step');
ylabel('Distance (in m)');
title('Average OSPA Evaluation');
