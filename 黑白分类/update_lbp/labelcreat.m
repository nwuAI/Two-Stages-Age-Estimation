clc;
clear all;
label = zeros(2*200,1);
a = zeros(200,1);
for t = 0:1
    a(:,1) = t;
    i = t * 200 + 1;
    j = 200 + (t*200);
    label(i:j,1) = a;
end
save LGP_label label -v7.3