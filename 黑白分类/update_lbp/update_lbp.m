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
imgn=zeros(m,n);%��ʼ���������
%ÿ��ѭ����ʼpowΪ0����ʮ��������ʱ��Ҫ�õ������Ϊ8
%��������ͼ��
for i=3:m-2   %Ҫ��������ؾ������
   for j=3:n-2%Ҫ��������ؾ������
       %������Ȧ
       x_center = 0;
       for cow = i-1 : i+1   %�����ĵ����λ�õ���
           for line = j-1 : j+1  %�����ĵ����λ�õ���
               if(cow~=i || line~=j)%�ų����ĵ�����ĵ�ĶԱ�
                   x_center = (x_center + (img(cow,line) - img(i,j)))/8;
               end
               
           end
       end
%        x_center = img(i,j);
       x_list = [];
       for compare_cow = i-1 : i+1           %��x_center��Ƚ�
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
%        %������Ȧ
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
       
       
% %        %�Ƚ�x_list��y_list��ֵ�������ͬΪ1����ͬΪ0
% %        center_list = [];
% %        for len=1:8
% %            center_list_word = (x_list(len) == y_list(len));
% %            center_list = [center_list,center_list_word];
% %        end
% %        
% % %        ���ȽϵĽ��������ת����ʮ���������滻���ĵ������ֵ
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



