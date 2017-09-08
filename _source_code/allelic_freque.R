
data(mice)
?mice
W <- mice.X

allelematrix<-matrix(1:2*10346,2,10346)

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
  allelematrix[1,i]<-(x*2+y)/(1814*2)
  allelematrix[2,i]<-(z*2+y)/(1814*2)
}

