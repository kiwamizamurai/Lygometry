
#install.packages("BGLR")
#library(BGLR)
data(mice)
?mice
W <- mice.X
head(W[1:3,1:5])
dim(W)
genomatrix<-matrix(1:3*10346,3,10346)

for(i in 1:dim(W)[2]){
  x<-0
  y<-0
  z<-0
  for(m in 1:dim(W)[1]){
    if(W[m,i]==0){
      x<-x+1
    }
    else if(W[m,i]==1){
      y<-y+1
    }
    else
      z<-z+1
  }
  genomatrix[1,i]<-x/1814
  genomatrix[2,i]<-y/1814
  genomatrix[3,i]<-z/1814
}

dim(genomatrix)
rownames(genomatrix)<-c("aa","Aa","AA")
genomatrix[1:3,1:3]
write.table(genomatrix,"~/Desktop/genotype_frequ.csv",quote=F,sep="\t", row.names=F, col.names=F)
