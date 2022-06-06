clc
clear
load('draw_loss.mat')
subplot('Position',[0.13,0.57,0.37,0.37]);
plot(step11,loss11);
hold on
y=fit1(x);
plot(x,y);
hold on
plot(step12,loss12,'.');
hold off
subplot('Position',[0.58,0.57,0.37,0.37]);
plot(step751,loss751);
hold on
y=fit2(x);
plot(x,y);
hold on
plot(step752,loss752,'.');
hold off
subplot('Position',[0.13,0.11,0.37,0.37]);
plot(step501,loss501);
hold on
y=fit3(x);
plot(x,y);
hold on
plot(step502,loss502,'.');
hold off
subplot('Position',[0.58,0.11,0.37,0.37]);
plot(step251,loss251);
hold on
y=fit4(x);
plot(x,y);
hold on
plot(step252,loss252,'.');
hold off
%设置坐标轴线宽1.0，数据线宽2.0，点大小15，字体Times New Romam，大小16号