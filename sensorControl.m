function [otp_sensor_pos] = sensorControl(current_pos, est_state, sensor_network)

    %disp('spacing');
    otp_sensor_pos = current_pos;

    avaiable_sensor = findNeighbour(current_pos, sensor_network);
    
    ospa_cutoff = 200;
    ospa_order = 2;
    
    if (~isempty(est_state))
        for row = 1 : size(avaiable_sensor, 1)
            for col = 1 : size(avaiable_sensor, 2)
                
                sensor_pos = [avaiable_sensor(row,col).x; avaiable_sensor(row,col).y];
                sensor_mat = repmat(sensor_pos, 1, size(est_state,2));

                ospa = ospa_dist(sensor_mat, est_state(1:2,:), ospa_cutoff, ospa_order);
                
                %disp([num2str(sensor_pos(1)), '  ', num2str(sensor_pos(2)), '  ', num2str(ospa)]);
                
                if (ospa < ospa_cutoff)
                    otp_sensor_pos = [avaiable_sensor(row,col).y/50 + 1, avaiable_sensor(row,col).x/50 + 1];
                end
            end
        end
    else
        otp_sensor_pos = current_pos;
    end
end

function [sensor_neighbourhood] = findNeighbour(current_pos, sensor_network)

    window_size = [3 3];

    shift_matrix = [(window_size(1) - 1)/2, (window_size(2) - 1)/2];
    
    r_start = max(1, current_pos(1) - shift_matrix(1));
    c_start = max(1, current_pos(2) - shift_matrix(2));

    r_end = min(current_pos(1) + shift_matrix(1), size(sensor_network,1));
    c_end = min(current_pos(2) + shift_matrix(2), size(sensor_network,2));

    sensor_neighbourhood = sensor_network(r_start : r_end, c_start : c_end);

end
