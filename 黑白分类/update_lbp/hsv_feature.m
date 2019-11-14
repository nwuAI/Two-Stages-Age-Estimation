clear;
clc;
hsv_fea = [];
stImageFilePath = 'C:\Users\12602\Desktop\update_lbp\color_datebase\';
dirImagePathList = dir(strcat(stImageFilePath,'*.jpg'));        %读取该文件夹下所有图片的路径（字符串格式）
iImageNum = length(dirImagePathList);  
if iImageNum > 0                                                %批量读入图片
    for t = 1 : iImageNum                                    %循环读取每个图片
        stImagePath   = dirImagePathList(t).name;
        mImageCurrent = imread(strcat(stImageFilePath,stImagePath));
        feature(t,:) = HSV(mImageCurrent);
    end
end
hsv_fea = [hsv_fea; feature];
save hsv_fea_result  hsv_fea  -v7.3 