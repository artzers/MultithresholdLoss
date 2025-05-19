
k=20;
x=10:0.5:50;
% threv = 20;
y_true = 30;
loss = zeros(1,81);
mseloss =  zeros(1,81);
i=1;
threv=[10,20,30,40];
% threv=[10,15,20,25,30,35,40];
% threv = threv + 5;
for y_pred =10:0.5:50
    for k=1:length(threv)
        loss(1,i) = loss(1,i)+ forward(k, y_pred, y_true, threv(k));
    end
    mseloss(1,i) = 10*(y_pred/50-y_true/50)^2;
i=i+1;
end
figure();
plot(x, loss);
hold on;
plot(x, mseloss);