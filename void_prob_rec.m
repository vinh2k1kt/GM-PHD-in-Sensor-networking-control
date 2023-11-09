function [otp_pos, min_void_probability, void_prob_matrix] = void_prob_rec(sensor_index, sensor_network, w, m, P, sur_area)
   
    avaiable_sensor = findNeighbour(sensor_index, sensor_network);
    
    min_void_probability = inf;
    
    void_prob_matrix = Inf(size(avaiable_sensor,1), size(avaiable_sensor,2));

    if (~isempty(m))
        for row = 1 : size(avaiable_sensor, 1)
            for col = 1 : size(avaiable_sensor, 2)

                integral = 0;

                sensor_pos = [avaiable_sensor(row,col).x, avaiable_sensor(row,col).y];
                
                % Rec Size [x_min y_min; x_max y_max] 
                rec_size = sur_area(2,:) / 20; 

                rec_bound = [-rec_size(:,1)/2, -rec_size(:,2)/2; rec_size(:,1)/2, rec_size(:,2)/2];
                
                % Shifting Rec_Coordinate By Sensor Pos
                rec_coordinate = [rec_bound(:,1) + sensor_pos(:,1), rec_bound(:,2) ...
                                  + sensor_pos(:,2)];
            
                % x_min <= x <= x_max
                % y_min <= y <= y_max
            
                x_step = 20;
                y_step = 20;
            
                delta_x = (rec_coordinate(2,1) - rec_coordinate(1,1)) / x_step;
                delta_y = (rec_coordinate(2,2) - rec_coordinate(1,2)) / y_step;
            
                delta_a = delta_x * delta_y;
                
                for x_i = 1 : x_step
                    for y_i = 1 : y_step
                        
                        x = rec_coordinate(1,1) + (x_i + 0.5) * delta_x;
                        y = rec_coordinate(1,2) + (y_i + 0.5) * delta_y;
            
                        r_ij = [x;y];

                        integral = integral + intensity(r_ij, w, m, P) * delta_a;
                    end
                end
                
                void_prob = exp(-integral);

                void_prob_matrix(row, col) = void_prob; 

                if (void_prob <= min_void_probability)

                    min_void_probability = void_prob;
                    
                    otp_pos = [avaiable_sensor(row,col).y/100 + 1, ...
                               avaiable_sensor(row,col).x/100 + 1]';
                end
            end
        end
    else
        otp_pos = sensor_index;
    end
end

function [intensity_val] = intensity(r_ij, w, m, P)

    % p(x|m,P) = N(m,P) =
    % (2pi)^(-D/2)*det(P)^(-1/2) * exp(-1/2*(x-m)' * P' * (x-m))

    D = size(r_ij,1); %Dimension;
    
    m = m(1:D,:);
    P = P(1:D,1:D,:);

    intensity_val = 0;

    for i = 1 : size(m,2)
        gaussian_distribution = w(i) * (2*pi)^(-D/2)*det(P(:,:,i))^(-0.5)*...
            exp(-0.5*(r_ij - m(:,i))' * 1/P(:,:,i) * (r_ij - m(:,i)));

        intensity_val = intensity_val + gaussian_distribution;
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