clear all;clc;tic;

model = gen_model();

%Plot pD profile
x1=0:1000;
mu=[500 500];

[X1,X2] = meshgrid(x1,x1);
F = 2*pi*sqrt(model.P_D_cov_det)*mvnpdf([X1(:) X2(:)],mu,model.P_D_cov);
F = reshape(F,length(x1),length(x1));
contour(x1,x1,F,[.06:.06:.96 .99 .999],'ShowText','on','LineWidth',1);
axis([0 1000 0 1000]); hold on;
sensor_plot = plot(mu(1),mu(2), 'MarkerFaceColor',[1 0.4784 0.4784], ...
                                'Marker','d', ...
                                'MarkerSize',10, ...
                                'MarkerFaceColor','c', ...
                                'MarkerEdgeColor','red');
axis square;
xlabel('x [m]');ylabel('y [m]');
set(gca,'FontSize',12);
set(findall(gcf,'type','line'),'linewidth',1.5);

h = legend(sensor_plot, ...
    'Sensor', ...
    'Location', 'northeast');
    set(h,'FontSize',15);
hold off;
