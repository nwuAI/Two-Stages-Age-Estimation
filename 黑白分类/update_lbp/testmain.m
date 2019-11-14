clear;
clc;
DLGP123 = [];
stImageFilePath = 'C:\Users\12602\Desktop\update_lbp\datebase\';
dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %读取该文件夹下所有图片的路径（字符串格式）
iImageNum = length(dirImagePathList);  
% str = strcat ('D:\闪闪论文\CMU-PIE_database\CMUPIE\',int2str(k),'\') ;
% stImageFilePath  = str;
% dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %读取该文件夹下所有图片的路径（字符串格式）
if iImageNum > 0                                                %批量读入图片
    for t = 1 : iImageNum                                    %循环读取每个图片
        stImagePath   = dirImagePathList(t).name;
        mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
        update_relbp_one(t,:) = update_lbp(mImageCurrent);
    end
end
DLGP123 = [DLGP123; update_relbp_one];
save DLGP123_result  DLGP123  -v7.3 