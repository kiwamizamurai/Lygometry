
MAF<-matrix(1:2*dim(allelematrix)[2],2,dim(allelematrix)[2])
rownames(MAF)<-c("Allele","frequency")

for(i in 1:dim(allelematrix)[2]){
  if(allelematrix[1,i] > allelematrix[2,i]){
    MAF[1,i]<-"A"
    MAF[2,i]<-allelematrix[2,i]
  }
  else
    MAF[1,i]<-"a"
    MAF[2,i]<-allelematrix[1,i]
}