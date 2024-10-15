clc; clear; close all;
LOOKAHEAD = 80;
YAW_SCALE = 0.18;

log = readmatrix('ztesting.csv');
[nr_xs, nr_ys, nr_zs, nr_yaws, nr_thrusts, nr_roll_rates, nr_pitch_rates, nr_yaw_rates, nr_des_xs, nr_des_ys, nr_des_zs, nr_des_yaws] = nr_fix_data3(log);

% Apply YAW_SCALE
nr_yaws = nr_yaws * YAW_SCALE;
nr_des_yaws = nr_des_yaws * YAW_SCALE;

% Trim the arrays with matching sizes
nr_xs_trimmed = nr_xs(LOOKAHEAD+1:end);
nr_des_xs_trimmed = nr_des_xs(LOOKAHEAD+1:end); % Match size to nr_xs_trimmed

nr_ys_trimmed = nr_ys(LOOKAHEAD+1:end);
nr_des_ys_trimmed = nr_des_ys(LOOKAHEAD+1:end); % Match size to nr_ys_trimmed

nr_zs_trimmed = nr_zs(LOOKAHEAD+1:end);
nr_des_zs_trimmed = nr_des_zs(LOOKAHEAD+1:end); % Match size to nr_zs_trimmed

nr_yaws_trimmed = nr_yaws(LOOKAHEAD+1:end);
nr_des_yaws_trimmed = nr_des_yaws(LOOKAHEAD+1:end); % Match size to nr_yaws_trimmed

% Calculate RMSE
err = (nr_xs_trimmed - nr_des_xs_trimmed).^2 + ...
      (nr_ys_trimmed - nr_des_ys_trimmed).^2 + ...
      (nr_zs_trimmed - nr_des_zs_trimmed).^2 + ...
      (nr_yaws_trimmed - nr_des_yaws_trimmed).^2;
rmse_value = sqrt(mean(err));
    
disp(['RMSE: ', num2str(rmse_value)]);


% Create figure and 4x3 tiled layout for subplots
figure;
tiledlayout(4, 3, 'TileSpacing', 'compact', 'Padding', 'compact');


% Row 1: Plot x, y, z, yaw vs references
% x vs x_ref
nexttile;
plot(nr_xs(LOOKAHEAD+1:end), 'r', 'DisplayName', 'x');
hold on;
plot(nr_des_xs(1:end-LOOKAHEAD), 'b--', 'DisplayName', 'x\_ref');
ylabel('x / x\_ref');
xlabel('time');
legend('show');
% xlim(x_lim);

% y vs y_ref
nexttile;
plot(nr_ys(LOOKAHEAD+1:end), 'r', 'DisplayName', 'y');
hold on;
plot(nr_des_ys(1:end-LOOKAHEAD), 'b--', 'DisplayName', 'y\_ref');
ylabel('y / y\_ref');
xlabel('time');
legend('show');
% xlim(x_lim);

% z vs z_ref
nexttile;
plot(-1*nr_zs(LOOKAHEAD+1:end), 'r', 'DisplayName', 'z');
hold on;
plot(-1*nr_des_zs(1:end-LOOKAHEAD), 'b--', 'DisplayName', 'z\_ref');
ylabel('z / z\_ref');
xlabel('time');
legend('show');
% xlim(x_lim);
ylim([-1*max(nr_zs)+0.1, 0]);

% yaw vs yaw_ref
nexttile;
plot(nr_yaws(LOOKAHEAD+1:end) * YAW_SCALE, 'r', 'DisplayName', 'yaw');
hold on;
plot(nr_des_yaws(1:end-LOOKAHEAD) * YAW_SCALE, 'b--', 'DisplayName', 'yaw\_ref');
ylabel('yaw / yaw\_ref');
xlabel('time');
legend('show');
% xlim(x_lim);

% Row 2: Plot throttle, roll_rate, pitch_rate, yaw_rate vs time
% throttle vs time
nexttile;
plot(nr_thrusts, 'b', 'DisplayName', 'throttle');
ylabel('Throttle');
xlabel('Time');
% ylim([-0.2, 1.2]);
% xlim(x_lim);
% yticks(-0.2:0.2:1.2);
legend('show');

% roll_rate vs time
nexttile;
plot(nr_roll_rates, 'b', 'DisplayName', 'roll\_rate');
hold on;
yline(0.8, 'r--', 'DisplayName', '+0.8');
yline(-0.8, 'r--', 'DisplayName', '-0.8');
ylabel('Roll Rate');
xlabel('Time');
ylim([-1.0, 1.0]);
% xlim(x_lim);
yticks(-1:0.2:1);
legend('show');

% pitch_rate vs time
nexttile;
plot(nr_pitch_rates, 'g', 'DisplayName', 'pitch\_rate');
hold on;
yline(0.8, 'r--', 'DisplayName', '+0.8');
yline(-0.8, 'r--', 'DisplayName', '-0.8');
ylabel('Pitch Rate');
xlabel('Time');
ylim([-1.0, 1.0]);
% xlim(x_lim);
yticks(-1:0.2:1);
legend('show');

% yaw_rate vs time
nexttile;
plot(nr_yaw_rates, 'b', 'DisplayName', 'yaw\_rate');
hold on;
yline(0.8, 'r--', 'DisplayName', '+0.8');
yline(-0.8, 'r--', 'DisplayName', '-0.8');
ylabel('Yaw Rate');
xlabel('Time');
ylim([-1.0, 1.0]);
% xlim(x_lim);
yticks(-1:0.2:1);
legend('show');

% Final adjustments
sgtitle('Comparison and Input Plots');


function [xs, ys, zs, yaws, thrusts, roll_rates, pitch_rates, yaw_rates, des_xs, des_ys, des_zs, des_yaws]=nr_fix_data3(M)
    MM = M(:,2);
    ind = ~isnan(MM);
    MMM=MM(ind);
    MMM(end+1) = 0;
    % MMM = MMM(2:end)
    num_rows = 13;
    numCols = length(MMM) / num_rows;
    tempMatrix = reshape(MMM, num_rows, numCols);
    tempMatrix(num_rows, :) = [];


    resultMatrix = tempMatrix(:,1:end);
    % resultMatrix = tempMatrix;
    resultMatrixSize = size(resultMatrix);
    numCols = resultMatrixSize(2);
    numRows = resultMatrixSize(1);    
    % resultMatrix = resultMatrix([2, 3:end, 1], :);

    xs=resultMatrix(1,:);
    ys=resultMatrix(2,:);
    zs=-resultMatrix(3,:)- (0.15)*ones(1,numCols); %-0.24679;;;
    yaws=resultMatrix(4,:);
    thrusts=-resultMatrix(5,:);
    roll_rates=resultMatrix(6,:);
    pitch_rates=resultMatrix(7,:);
    yaw_rates=resultMatrix(8,:);
    des_xs=resultMatrix(9,:);
    des_ys=resultMatrix(10,:);
    des_zs=-resultMatrix(11,:);
    des_yaws=resultMatrix(12,:);

    
end