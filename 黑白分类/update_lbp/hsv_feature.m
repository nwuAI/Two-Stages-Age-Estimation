clear;
clc;
hsv_fea = [];
stImageFilePath = 'C:\Users\12602\Desktop\update_lbp\color_datebase\';
dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %��ȡ���ļ���������ͼƬ��·�����ַ�����ʽ��
iImageNum = length(dirImagePathList);  
if iImageNum > 0                                                %��������ͼƬ
    for t = 1 : iImageNum                                    %ѭ����ȡÿ��ͼƬ
        stImagePath   = dirImagePathList(t).name;
        mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
        feature(t,:) = HSV(mImageCurrent);
    end
end
hsv_fea = [hsv_fea; feature];
save hsv_fea_result  hsv_fea  -v7.3 