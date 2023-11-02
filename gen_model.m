function model = gen_model

% Transition Model
model.x_dim = 4;     % x dimension
model.z_dim = 2;     % z dimension
model.dt = 10;       % sampling period

% Surviving/Death Parameters
model.P_S = 0.99;     % Surviving Probability

% Detection/Miss Detection Parameters
model.P_D = 0.99;        % probability of detection in measurements
model.P_MD= 1-model.P_D; % probability of missed detection in measurements

model.F = [1 0 model.dt 0; 0 1 0 model.dt; 0 0 1 0; 0 0 0 1]; % Transition Model
model.H = [1 0 0 0; 0 1 0 0];                                 % Measurement Model
model.Q = [3 0 0 0; 0 2.5 0 0; 0 0 2 0; 0 0 0 1];             % Transiion Noise Covariance
model.R = 15 * [.5 0; 0 .65];                                 % Measurement Noise Covariance

% Birth parameters
model.L_birth = 4;
model.w_birth= zeros(model.L_birth,1);                                %weights of Gaussian birth terms (per duration)
model.m_birth= zeros(model.x_dim,model.L_birth);                      %means of Gaussian birth terms 
model.P_birth= zeros(model.x_dim,model.x_dim,model.L_birth);          %cov of Gaussian birth terms

model.w_birth= repmat(.01, model.L_birth, 1);                                            %birth of simulation (example 1)
model.m_birth= [ [0; 250; 0; 0] [0; 750; 0; 0] [250; 0; 0; 0] [750; 0; 0; 0]];

model.P_birth(:,:,:)= repmat(diag([200^2, 200^2, 25, 25]),1,1,model.L_birth);

% Clutter parameters
model.lambda_c = 20;                                        % clutter rate
model.range_c= [-1000 1000; -1000 1000];                    % uniform clutter region
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); % uniform clutter density