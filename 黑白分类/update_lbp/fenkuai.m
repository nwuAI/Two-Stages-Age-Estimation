function Fenkuai=fenkuai(I)
%I=imread('lena.jpg');
heights=size(I,1);  % 图像的高
widths=size(I,2);  % 图像的宽
m=2; % 假设纵向分成6幅图
n=2; % 假设横向分成6幅图
% 考虑到rows和cols不一定能被m和n整除，所以对行数和列数均分后要取整
rows=round(linspace(0,heights,m+1)); % 各子图像的起始和终止行标
cols=round(linspace(0,widths,n+1)); % 各子图像的起始和终止列标
i=0;
blocks=cell(m,n);  % 用一个单元数组容纳各个子图像
for k1=1:m
    for k2=1:n
        blocks{k1,k2}=I(rows(k1)+1:rows(k1+1),cols(k2)+1:cols(k2+1),:);
        subimage=blocks{k1,k2};
        % 以下是对subimage进行处理
       
%         subimage=cslbp_1(subimage);
        % 以上是对subimage进行处理
%         blocks{k1,k2}=subimage;
%          i=i+1;
%           subplot(m,n,i),imshow(blocks{k1,k2},[]);
    end
end
Fenkuai=blocks;
% processed=I; % processed为处理后的图像，用原图像对其初始化
% % % 以下为拼接图像
% % for k1=1:m
% %     for k2=1:n
% %         processed(rows(k1)+1:rows(k1+1),cols(k2)+1:cols(k2+1),:)=blocks{k1,k2};
% %     end
% % end
% figure,imshow(processed,[]);
% Fenkuai=processed;
