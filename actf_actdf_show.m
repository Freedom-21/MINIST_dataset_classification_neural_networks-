% implement actf & check it on a graph
x = 0:0.01:5;
figure;
hold on;
plot(x, actf_ref(x));
plot(x, actdf_ref(actf_ref(x)));
hold off;
disp('finish!!!')