function [m_predict, P_predict] = predict_KF(model, m, P)
    % x: State Before Update
    % P: Covariance Matrix Before Update
    % F: Transition Matrix
    % Q: Transition Noise Covariance Matrix

    plength= size(m,2);

    m_predict = zeros(size(m));
    P_predict = zeros(size(P));

    for idx = 1:plength
        [m_temp,P_temp] = predict_single(model.F, model.Q, m(:,idx), P(:,:,idx));
        m_predict(:,idx) = m_temp;
        P_predict(:,:,idx) = P_temp;
    end
end

function [m_predict,P_predict] = predict_single(F,Q,m,P)
    % Predict New State
    m_predict = F * m;

    % Predict New Covariance Matrix
    P_predict = Q + F * P * F';
end