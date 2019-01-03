datestr(now,'dd-mm-yyyy HH:MM:SS FFF')
dic = 'C:\Users\Wanting\Documents\MATLAB\5243\project 3\images2';
numpic = 3000;

% Parameters:
clear param

param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 6;
param.fc_prefilt = 6;

T = table;

for i = 1:numpic
    filename = fullfile ( dic, sprintf('%04d.jpg',i) );
    img0 = imread(filename);
    [gist, param] = LMgist(img0, '', param);
    T(i,:) = table(gist);
end

writetable(T, 'gist6.csv');
datestr(now,'dd-mm-yyyy HH:MM:SS FFF')