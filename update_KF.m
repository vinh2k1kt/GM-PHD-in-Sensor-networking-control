function [m_update, P_update] = update_KF(z, model, m, P)

    plength= size(m,2);
    zlength= size(z,2);

    m_update = zeros(model.x_dim,plength,zlength);
    P_update = zeros(model.x_dim,model.x_dim,plength);
    for idx = 1:plength
        [m_temp, P_temp] = update_single(z,model.H,model.R,m(:,idx),P(:,:,idx));
        m_update(:,idx,:) = m_temp;
        P_update(:,:,idx) = P_temp;
    end
end

function [m_temp,P_temp] = update_single(z,H,R,m,P)
    % Caculate predicted measurement;
    mu = H * m;

    % Calculate residual
    residual = z - repmat(mu,[1 size(z,2)]);

    % Calculate Innovation Matrix
    S = R + H * P * H';
    Vs = chol(S); 
    inv_sqrt_S = inv(Vs);
    iS = inv_sqrt_S * inv_sqrt_S';
    det_S= prod(diag(Vs))^2;

    % Calculate Kalman Gain
    K = P * H' * iS;

    % Update
    m_temp = repmat(m,[1 size(z,2)]) + K * residual;
    P_temp = (eye(size(P)) - K * H) * P;
end