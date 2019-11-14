function result =update_lbp(img )
% img=imread('2.bmp');
% img=rgb2gray(img);
result = [];
seg =fenkuai(img);
[max_row,max_col] = size(seg);
for x = 1:max_row;
    for y = 1:max_col
        img = seg{x,y};
[m,n]=size(img);
imgn=zeros(m,n);%初始化结果矩阵
%每次循环开始pow为0，求十进制数字时需要用到，最大为8
%遍历整幅图像
for i=3:m-2   %要计算的像素矩阵的行
   for j=3:n-2%要计算的像素矩阵的列
       %计算里圈
       x_center = 0;
       for cow = i-1 : i+1   %与中心点相对位置的行
           for line = j-1 : j+1  %与中心点相对位置的列
               if(cow~=i || line~=j)%排除中心点和中心点的对比
                   x_center = (x_center + (img(cow,line) - img(i,j)))/8;
               end
               
           end
       end
%        x_center = img(i,j);
       x_list = [];
       for compare_cow = i-1 : i+1           %和x_center相比较
           for compare_line = j-1 : j+1
               if(compare_cow~=i || compare_line~=j)
                   x_word = img(compare_cow,compare_line) >= img(i,j);
                   x_list = [x_list,x_word];
               end
               
           end
       end

%        sum = 0;
%        for center_i = 1:8
%            sum = sum + x_list(center_i) * (2^(8-center_i));
%        end
%        imgn(i,j) = sum;
%        
%        %计算外圈
       y_center = 0;
       for y_cow = i-2 :2: i+2
           for y_line = j-2 :2: j+2
               if(y_cow~=i || y_line~=j)
                   y_center = (y_center + (img(y_cow,y_line) - img(i,j)))/8;
               end
           end
       end
       
       y_list = [];
       for y_compare_cow = i-2 :2: i+2
           for y_compare_line = j-2 :2: j+2
               
               if(y_compare_cow~=i || y_compare_line~=j)
                   y_word = img(y_compare_cow,y_compare_line) >= img(i,j);
                   y_list = [y_list,y_word];
               end
               
           end
       end
      
       sum_in = 0;
       sum_out = 0;
       for center_i = 1:8
           sum_in = sum_in + x_list(center_i) * (2^(8-center_i));
           sum_out = sum_out + y_list(center_i) * (2^(8-center_i));
       end
       imgn(i,j) = sum_in * 0.6 + sum_out * 0.4;
       
       
% %        %比较x_list和y_list的值，如果相同为1，不同为0
% %        center_list = [];
% %        for len=1:8
% %            center_list_word = (x_list(len) == y_list(len));
% %            center_list = [center_list,center_list_word];
% %        end
% %        
% % %        将比较的结果二进制转化成十进制数，替换中心点的像素值
% %        sum = 0;
% %        for center_i = 1:8
% %            sum = sum + center_list(center_i) * (2^(8-center_i));
% %        end
       
% %        imgn(i,j) = sum;
   end
end
imgn = reshape(imgn,[1,size(imgn,1)*size(imgn,2)]);
% imgn_y = histogram(imgn,100);
% counts = imgn_y.Values;
result = [result,imgn];
    end
end



