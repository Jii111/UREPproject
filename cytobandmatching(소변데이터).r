## cytoband_hg19.txt 사용
library(dplyr);library(readr);library(tidyr)

cyto <- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/0222_hg19시도/cytoBand_hg19.txt", sep = '\t')
colnames(cyto) <- c("chr","start","end","cyto","gieStain");
cyto$Chromosome <- substr(cyto$chr, 4, 10000);
cyto$name <- paste0("cyto.",cyto$Chromosome, cyto$cyto);
p <- length(cyto$name)

disease<- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/CNV_Classifier2021/140cases/BLCA_100k.txt", sep = '\t', header=TRUE);
colnames(disease)[1] <- 'Chrom'

normal<- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/CNV_Classifier2021/140cases/NL_100k.txt", sep = '\t', header=TRUE);
colnames(normal)[1] <- 'Chrom'


## cytoband matching
## blca
n <- nrow(disease); n
disease$cyto <- NA; sum(is.na(disease$cyto))

for(i in 1:n){
  
  #chromosome
  temp.i <- cyto[cyto$Chromosome == disease$Chrom[i],]
  #start, end
  where <- (temp.i$start <= disease$start[i] & temp.i$end >= disease$end[i])
  
  if(sum(where) == 0) {print(paste0('cyto NA index :', i))}
  else {disease$cyto[i] <- temp.i$name[where]}
}

colSums(is.na(disease))

## normal
n <- nrow(normal); n
normal$cyto <- NA; sum(is.na(normal$cyto))

for(i in 1:n){
  temp.i <- cyto[cyto$Chromosome == normal$Chrom[i],]
  where <- (temp.i$start <= normal$start[i] & temp.i$end >= normal$end[i])
  
  if(sum(where) == 0) {print(paste0('cyto NA index :', i))}
  else {normal$cyto[i] <- temp.i$name[where]}
}

colSums(is.na(normal))

## dataframe spread
## blca
colnames(disease)
disease2<-disease[,c(4:46)] %>% gather(id, Value,X350,X351,X376,X381, 
                                       X399,X409,X433,X438,X463,X473,X487, 
                                       X490,X491,X502,X510,X516,X517,X523, 
                                       X531,X535,X538,X545,B001,B002,B003, 
                                       B004,B005,B006,Q6.U,Q7.U,UP001,UP003,
                                       UP007,UP010,UP011,UP019,UP020,UP022,UP028,
                                       UP029,UP030,UP033) %>% 
  group_by(cyto,id) %>% summarise(m = mean(Value,na.rm=T)) %>% spread(cyto,m)
dim(disease2)

## normal
normal[,4:32]
colnames(normal)
normal2<-normal[,c(4:32)] %>% gather(id, Value,X416,X419,X481,
                                     X482,X492,X493,X500,X507,
                                     X508,X509,X512,X513,X518,X519,X521,
                                     X527,X528, X530,N001,N003,N004,UP015,UP016,
                                     UP023,UP024,UP026,UP031,UP032) %>% 
  group_by(cyto,id) %>% summarise(m = mean(Value,na.rm=T)) %>% spread(cyto,m)
dim(normal2)

# export data
write.csv(disease2,'C:/Users/judy0/OneDrive/바탕 화면/urep/0222_hg19_cytobandmatching/BLCA_100k_cyto_hg19.csv',row.names = T)
write.csv(normal2,'C:/Users/judy0/OneDrive/바탕 화면/urep/0222_hg19_cytobandmatching//NL_100k_cyto_hg19.csv',row.names = T)

