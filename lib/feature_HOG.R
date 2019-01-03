setwd("/Users/JHY/Documents/2018SpringCourse/Applied Data Science/Spring2018-Project3-Group1")
img_dir <- "./data/train/images/"
#source("http://bioconductor.org/biocLite.R")
#biocLite("EBImage")

feature_HOG<-function(img_dir){
  ### HOG: calculate the Histogram of Oriented Gradient for an image
  
  ### Input: a directory that contains images ready for processing
  ### Output: an .RData file contains features for the images
  
  ### load libraries
  library("EBImage")
  library("OpenImageR")
  
  dir_names <- list.files(img_dir)
  n_files <- length(dir_names)
  
  ### calculate HOG of images
  dat <- vector()
  for(i in 1:n_files){
    img <- readImage(paste0(img_dir,dir_names[i]))
    img<-rgb_2gray(img)
    dat<- rbind(dat,HOG(img))
  }
  
  ### output constructed features
  save(dat, file="./output/features/HOG.RData")
  return(dat)
}

dat_HOG<-feature_HOG(img_dir)
