% function loss_per_t = forward(k, y_pred, y_true, threv)
%     % 计算每个阈值下的概率
%     p = 1 ./ (1 + exp(-k * (y_pred - threv))); % [Batch, T,1,H,W]
% 
%     % 真实标签转换为多阈值二分类
%     y_true_t = double(y_true >= threv); % [Batch, T,1,H,W]
% 
%     % 计算各阈值TP, FP, FN
%     tp = sum(sum(y_true_t .* p, 'all')); % [T]
%     fp = sum(sum((1 - y_true_t) .* p, 'all')); % [T]
%     fn = sum(sum(y_true_t .* (1 - p), 'all')); % [T]
% 
%     % 各阈值CSI损失
%     eps = obj.eps;
%     csi_per_t = (tp + eps) ./ (tp + fp + fn + eps);
%     loss_per_t = 1 - csi_per_t;
% end
% 

function loss_per_t = forward(k, y_pred, y_true, threv)
    % 计算每个阈值下的概率
    p = 1 ./ (1 + exp(-k * (y_pred - threv))); % [Batch, T,1,H,W]

    % 真实标签转换为多阈值二分类
    y_true_t = double(y_true >= threv); % [Batch, T,1,H,W]

    % 计算各阈值TP, FP, FN
    tp = sum(sum(y_true_t .* p, 'all')); % [T]
    fp = sum(sum((1 - y_true_t) .* p, 'all')); % [T]
    fn = sum(sum(y_true_t .* (1 - p), 'all')); % [T]
    
%     y_true_t .* p + p-y_true_t .* p+y_true_t-y_true_t*p
%     p+y_true_t-y_true_t*p
%     p(1-y_true_t) + y_true_t
%     (y_true_t .* p)/[p(1-y_true_t) + y_true_t]
% p/y_true_t=p,->loss_per_t=1-p
    % 各阈值CSI损失
    eps = 1e-1;%0;%1e-6;
    csi_per_t = (tp + eps) ./ (tp + fp + fn + eps);
    loss_per_t = 1 - csi_per_t;
end




