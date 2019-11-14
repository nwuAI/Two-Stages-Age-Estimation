% RGB=imread('greens.jpg'); %将图像格式文件读入为MATLAB图像对象数组数据;
% 16*4*4, 这样提取出来的HSV特征是256维的;

function m=HSV(RGB)

[M,N,O] = size(RGB);
[h,s,v] = rgb2hsv(RGB);
H = h; S = s; V = v;
h = h*360;
%H量化为16级 S量化为4级 V量化为4级
for i = 1:M
    for j = 1:N
        if h(i,j)<=15||h(i,j)>345
            H(i,j) = 0;
        end
        if h(i,j)<=25&&h(i,j)>15
            H(i,j) = 1;
        end
        if h(i,j)<=45&&h(i,j)>25
            H(i,j) = 2;
        end
        if h(i,j)<=55&&h(i,j)>45
            H(i,j) = 3;
        end
        if h(i,j)<=80&&h(i,j)>55
            H(i,j) = 4;
        end
        if h(i,j)<=108&&h(i,j)>80
            H(i,j) = 5;
        end
        if h(i,j)<=140&&h(i,j)>108
            H(i,j) = 6;
        end
        if h(i,j)<=165&&h(i,j)>140
            H(i,j) = 7;
        end
        if h(i,j)<=190&&h(i,j)>165
            H(i,j) = 8;
        end
        if h(i,j)<=220&&h(i,j)>190
            H(i,j) = 9;
        end
        if h(i,j)<=255&&h(i,j)>220
            H(i,j) = 10;
        end
        if h(i,j)<=275&&h(i,j)>255
            H(i,j) = 11;
        end
        if h(i,j)<=290&&h(i,j)>275
            H(i,j) = 12;
        end
        if h(i,j)<=316&&h(i,j)>290
            H(i,j) = 13;
        end
        if h(i,j)<=330&&h(i,j)>316
            H(i,j) = 14;
        end
        if h(i,j)<=345&&h(i,j)>330
            H(i,j) = 15;
        end
    end
end
for i = 1:M
    for j = 1:N
        if s(i,j)<=0.15&&s(i,j)>0
            S(i,j) = 0;
        end
        if s(i,j)<=0.4&&s(i,j)>0.15
            S(i,j) = 1;
        end
        if s(i,j)<=0.75&&s(i,j)>0.4
            S(i,j) = 2;
        end
        if s(i,j)<=1&&s(i,j)>0.75
            S(i,j) = 3;
        end
    end
end
for i = 1:M
    for j = 1:N
        if v(i,j)<=0.15&&v(i,j)>0
            V(i,j) = 0;
        end
        if v(i,j)<=0.4&&v(i,j)>0.15
            V(i,j) = 1;
        end
        if v(i,j)<=0.75&&v(i,j)>0.4
            V(i,j) = 2;
        end
        if v(i,j)<=1&&v(i,j)>0.75
            V(i,j) = 3;
        end
    end
end
for  i = 1:M
    for j = 1:N
        L(i,j) = H(i,j)*16+S(i,j)*4+V(i,j); %归一化
    end
end
for i = 0:255
    HSVHist(i+1) = size(find(L==i),1);
end
m=HSVHist/sum(HSVHist);
% 将数据m保存到txt文件中
% fid = fopen('data_m.txt','w');
% fprintf(fid,'%f ',m);
% fclose(fid);
