function [likelihood] = cal_likelihood(z, model, m, P)

    plength= size(m,2);
    zlength= size(z,2);
    likelihood= zeros(plength,zlength);

    for idx = 1:plength

        % Caculate predicted Measurement
        mu = model.H * m(:,idx);

        % Calculate Innovation Matrix
        S = model.R + model.H * P(:,:,idx) * model.H';
        Vs = chol(S);
        inv_sqrt_S = inv(Vs);
        iS = inv_sqrt_S * inv_sqrt_S';
        det_S= prod(diag(Vs))^2;

        % Calculate Measurement Likelihood
        % pdf = [(2*pi)^(-N/2)*det(S)^(-1/2)] * exp(-1/2 * (z-m)' * 1/S * (z-m))
        % log(pdf) = -0.5*N*log(2*pi) - 0.5*log(det(S)) - (0.5)*(z-m)'*1/S*(z-m)
        % exp(log(pdf)) = pdf
        
        likelihood(idx,:) = exp(-0.5*size(z,1)*log(2*pi) - 0.5*log(det_S) - ...
            0.5*dot((z-repmat(mu,[1 size(z,2)])),iS*(z-repmat(mu,[1 size(z,2)]))))';
    end
end