clear all
close all
clc

performance = zeros(100,2);
count =1;
for ii = 10:10:1000

% Load img Information from data
galleryPath = 'C:\My Drive\Study Material\EMI\Project 2\GallerySet\';
probePath = 'C:\My Drive\Study Material\EMI\Project 2\ProbeSet\';
galleryset = dir(galleryPath);
probeset = dir(probePath);

% Change PCA dimensions 
pcaDimen = ii;

imgCount = 0;
for i =1:size(galleryset,1)
    if not(strcmp(galleryset(i).name,'.')|strcmp(galleryset(i).name,'..')|strcmp(galleryset(i).name,'Thumbs.db'))
        imgCount = imgCount +1;
    end
end

imgMatrix = [];
for j = 1:imgCount
    name = strcat(galleryPath,'\subject',int2str(j),'_img1.pgm');
    img = imread(name);
    img = im2double(img);
    [r c] = size(img);
    tmp = reshape(img,r*c,1)';
    imgMatrix(j,:) = tmp;
end

matMean = mean(imgMatrix);

meanCenteredMat = zeros(size(imgMatrix,1), size(imgMatrix,2));
for k = 1: size(meanCenteredMat,1)
    meanCenteredMat(k,:) = imgMatrix(k,:) - matMean;
end

covariance = cov(meanCenteredMat);
[vector,val] = eig(covariance);

eigenFaces = vector(:,size(vector,2)-pcaDimen : size(vector,2));
projectionMatrix = meanCenteredMat * eigenFaces;

% Testing Algorithm

testimgCount = 0;
for i =1:size(probeset,1)
    if not(strcmp(probeset(i).name,'.')|strcmp(probeset(i).name,'..')|strcmp(probeset(i).name,'Thumbs.db'))
        testimgCount = testimgCount +1;
    end
end


testimgMatrix1 = [];
testimgMatrix2 = [];
for j = 1:testimgCount/2
    name = strcat(probePath,'\subject',int2str(j),'_img2.pgm');
    img = imread(name);
    img = im2double(img);
    [r c] = size(img);
    tmp = reshape(img,r*c,1)';
    testimgMatrix1(j,:) = tmp;
    
    name = strcat(probePath,'\subject',int2str(j),'_img3.pgm');
    img = imread(name);
    img = im2double(img);
    [r c] = size(img);
    tmp = reshape(img,r*c,1)';
    testimgMatrix2(j,:) = tmp;
end

testMatMean1 = mean(testimgMatrix1+testimgMatrix2);
testMatMean2 = mean(testimgMatrix1+testimgMatrix2);

a=0;
b=0;
for k = 1:size(testimgMatrix1,1)
    testImg = (testimgMatrix1(k,:) - testMatMean1) * eigenFaces;
    
    euclide_dist = [ ];
    for l = 1:size(projectionMatrix,1)
        temp = sum((testImg - projectionMatrix(l,:)).^2).^0.5;
%         (norm(testImg-projectionMatrix(l,:)))^2;
        euclide_dist = [euclide_dist temp];
    end
    [euclide_dist_min, recognized_index] = min(euclide_dist);
    if recognized_index == k
        a=a+1;
    else
        b=b+1;
    end
end

for k = 1:size(testimgMatrix2,1)
    testImg = (testimgMatrix2(k,:)-testMatMean2) * eigenFaces;
    
    euclide_dist = [ ];
    for l = 1:size(projectionMatrix,1)
        temp = sum((testImg - projectionMatrix(l,:)).^2).^0.5;
%         (norm(testImg-projectionMatrix(l,:)))^2;
        euclide_dist = [euclide_dist temp];
    end
    [euclide_dist_min, recognized_index] = min(euclide_dist);
    if recognized_index == k
        a=a+1;
    else
        b=b+1;
    end
end

performance(count,2) = a/2;
performance(count,1) = ii;

count = count +1;
end


hold on
plot(performance(:,1), performance(:,2));
% scatter(performance(:,1), performance(:,2));
title('Performance of Facial Recognition System in terms of Recognition Rate');
xlabel('Number of EigenFaces');
ylabel('Recognition Rate (Accuracy of System)');