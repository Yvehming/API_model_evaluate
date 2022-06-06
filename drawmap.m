clc
clear
load('draw_map.mat')
subplot('Position',[0.13,0.57,0.37,0.37]);
y=fit1(x);
plot(x,y);
hold on
plot(step11,map11,'.');
hold off
subplot('Position',[0.58,0.57,0.37,0.37]);
y=fit2(x);
plot(x,y);
hold on
plot(step2,map2,'.');
hold off
subplot('Position',[0.13,0.11,0.37,0.37]);
y=fit3(x);
plot(x,y);
hold on
plot(step3,map3,'.');
hold off
subplot('Position',[0.58,0.11,0.37,0.37]);
y=fit4(x);
plot(x,y);
hold on
plot(step4,map4,'.');
hold off