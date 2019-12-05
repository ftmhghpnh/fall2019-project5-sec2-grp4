train_dir <- "../data/train/" 
train_image_dir <- paste(train_dir, "0/", sep="")
img_dir <- train_image_dir

feature <- function(img_dir, export=T){
  ### Sample simple feature: Extract row average raw pixel values as features
  ### load libraries
  library("EBImage")
  library(grDevices)
  
  ### Define the number of R, G and B
  nR <- 10
  nG <- 12
  nB <- 12 
  rBin <- seq(0, 1, length.out=nR)
  gBin <- seq(0, 1, length.out=nG)
  bBin <- seq(0, 1, length.out=nB)
  mat=array()
  freq_rgb=array()
  n_files <- length(list.files(img_dir))
  rgb_feature=matrix(nrow=n_files, ncol=nR*nG*nB)
  
  
  ########extract RGB features############

  dir_names <- list.files(img_dir)
  
  for (i in 1:n_files){
    mat <- imageData(readImage(paste0(img_dir,"/",dir_names[i])))
    mat_as_rgb <-array(c(mat,mat,mat),dim = c(nrow(mat),ncol(mat),3))
    freq_rgb <- as.data.frame(table(factor(findInterval(mat_as_rgb[,,1], rBin), levels=1:nR), 
                                    factor(findInterval(mat_as_rgb[,,2], gBin), levels=1:nG),
                                    factor(findInterval(mat_as_rgb[,,3], bBin), levels=1:nB)))
    rgb_feature[i,] <- as.numeric(freq_rgb$Freq)/(ncol(mat)*nrow(mat)) # normalization
    
    mat_rgb <-mat_as_rgb
    dim(mat_rgb) <- c(nrow(mat_as_rgb)*ncol(mat_as_rgb), 3)
  }
  
  
  rgb_feature <- cbind(rgb_feature,info$emotion_idx)
  
  if(export){
    write.csv(rgb_feature, file = "../output/rgb_0.csv")
  }
  return(data.frame(rgb_feature))
}
