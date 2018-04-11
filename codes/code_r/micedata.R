install.packages("BGLR")
library(BGLR)
data(mice)
?mice
W <- mice.X
head(W[1:3, 1:5])
dim(W) # 1814 10346
o <- data.frame(W)

library(sneer)
sneer(o,method='tsne',ndim=2)

?mice
o


