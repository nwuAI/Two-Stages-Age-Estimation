function Fenkuai=fenkuai(I)
%I=imread('lena.jpg');
heights=size(I,1);  % ͼ��ĸ�
widths=size(I,2);  % ͼ��Ŀ�
m=2; % ��������ֳ�6��ͼ
n=2; % �������ֳ�6��ͼ
% ���ǵ�rows��cols��һ���ܱ�m��n���������Զ��������������ֺ�Ҫȡ��
rows=round(linspace(0,heights,m+1)); % ����ͼ�����ʼ����ֹ�б�
cols=round(linspace(0,widths,n+1)); % ����ͼ�����ʼ����ֹ�б�
i=0;
blocks=cell(m,n);  % ��һ����Ԫ�������ɸ�����ͼ��
for k1=1:m
    for k2=1:n
        blocks{k1,k2}=I(rows(k1)+1:rows(k1+1),cols(k2)+1:cols(k2+1),:);
        subimage=blocks{k1,k2};
        % �����Ƕ�subimage���д���
       
%         subimage=cslbp_1(subimage);
        % �����Ƕ�subimage���д���
%         blocks{k1,k2}=subimage;
%          i=i+1;
%           subplot(m,n,i),imshow(blocks{k1,k2},[]);
    end
end
Fenkuai=blocks;
% processed=I; % processedΪ������ͼ����ԭͼ������ʼ��
% % % ����Ϊƴ��ͼ��
% % for k1=1:m
% %     for k2=1:n
% %         processed(rows(k1)+1:rows(k1+1),cols(k2)+1:cols(k2+1),:)=blocks{k1,k2};
% %     end
% % end
% figure,imshow(processed,[]);
% Fenkuai=processed;
