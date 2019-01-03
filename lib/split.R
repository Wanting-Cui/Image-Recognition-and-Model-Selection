sift <- read.csv("SIFT_train.csv", header = FALSE)[, -1]
index <- c(data.matrix(read.table("../TEST-NUMBER.txt")))
load("HOG.RData")
hog <- dat
lbp <- read.csv("lbp.csv", header = FALSE)
label <- read.csv("label_train.csv")

sift1 <- sift[index, ]
write.csv(sift1, file = "SIFT_test1.csv", row.names = FALSE)

sift2 <- sift[-index, ]
write.csv(sift2, file = "SIFT_train1.csv", row.names = FALSE)

lbp1 <- lbp[index, ]
write.csv(lbp1, file = "lbp_test1.csv", row.names = FALSE)

lbp2 <- lbp[-index, ]
write.csv(lbp2, file = "lbp_train1.csv", row.names = FALSE)

hog1 <- hog[index, ]
save(hog1, file = "hog_test1.RData")

hog2 <- hog[-index, ]
save(hog2, file = "hog_train.RData")

label1 <- label[index, 3]
write.csv(label1, file = "label_test1.csv", row.names = FALSE)

label2 <- label[-index, 3]
write.csv(label2, file = "label_train1,csv", row.names = FALSE)
