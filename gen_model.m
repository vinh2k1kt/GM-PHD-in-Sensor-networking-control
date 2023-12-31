function model = gen_model

% Transition Model
model.x_dim = 4;     % x dimension
model.z_dim = 2;     % z dimension
model.dt = 1;       % sampling period

% Surviving/Death Parameters
model.P_S = 0.99;     % Surviving Probability

% Detection/Miss Detection Parameters
model.P_D = 0.99;        % probability of detection in measurements
model.P_MD= 1-model.P_D; % probability of missed detection in measurements

%model.P_D_cov = 1.5e5*[5 -4;-4 6];
model.P_D_cov = 1e5*[3 -2.4;-2.4 3.6];
model.P_D_cov_inv = inv(model.P_D_cov);
model.P_D_cov_det = model.P_D_cov(1,1)*model.P_D_cov(2,2)-...
model.P_D_cov(1,2)*model.P_D_cov(2,1);

model.F = [1 0 model.dt 0; 0 1 0 model.dt; 0 0 1 0; 0 0 0 1]; % Transition Model
model.H = [1 0 0 0; 0 1 0 0];                                 % Measurement Model
model.Q = 27*[model.dt^3 0 model.dt^2/54 0;                   % Transiion Noise Covariance
              0 model.dt^3 0 model.dt^2/54; 
              model.dt^2/54 0 model.dt/81 0; 
              0 model.dt^2/54 0 model.dt/81];             
model.R = [9 0; 0 9];                                  % Measurement Noise Covariance

% Birth parameters
model.L_birth = 1;
model.w_birth= zeros(model.L_birth,1);                                %weights of Gaussian birth terms (per duration)
model.m_birth= zeros(model.x_dim,model.L_birth);                      %means of Gaussian birth terms 
model.P_birth= zeros(model.x_dim,model.x_dim,model.L_birth);          %cov of Gaussian birth terms

model.w_birth= repmat(.005, model.L_birth, 1);                                            %birth of simulation (example 1)
model.m_birth= [950;200; -2; 1];

model.P_birth(:,:,:)= repmat(diag([100, 100, 25, 25]),1,1,model.L_birth);

% Clutter parameters
model.lambda_c = 20;                                        % clutter rate
model.range_c= [-1000 1000; -1000 1000];                    % uniform clutter region
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); % uniform clutter density