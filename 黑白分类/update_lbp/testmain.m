clear;
clc;
DLGP123 = [];
stImageFilePath = 'C:\Users\12602\Desktop\update_lbp\datebase\';
dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %��ȡ���ļ���������ͼƬ��·�����ַ�����ʽ��
iImageNum = length(dirImagePathList);  
% str = strcat ('D:\��������\CMU-PIE_database\CMUPIE\',int2str(k),'\') ;
% stImageFilePath  = str;
% dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %��ȡ���ļ���������ͼƬ��·�����ַ�����ʽ��
if iImageNum > 0                                                %��������ͼƬ
    for t = 1 : iImageNum                                    %ѭ����ȡÿ��ͼƬ
        stImagePath   = dirImagePathList(t).name;
        mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
        update_relbp_one(t,:) = update_lbp(mImageCurrent);
    end
end
DLGP123 = [DLGP123; update_relbp_one];
save DLGP123_result  DLGP123  -v7.3 