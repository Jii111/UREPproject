library(dplyr);library(readr);library(tidyr)

blca_tcga <- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/CNV_Classifier2021/BLCA/BLCA_disease.txt",header=TRUE)
cyto <- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/0222_hg19_cytobandmatching/cytoBand_hg19.txt", sep = '\t');
colnames(cyto) <- c("chr","start","end","cyto","gieStain");
cyto$Chromosome <- substr(cyto$chr, 4, 10000);
cyto$name <- paste0("cyto.",cyto$Chromosome, cyto$cyto);
p <- length(cyto$name)

# colnames :  "Sample" "Chromosome" "Start" "End" "Num_Probes" "Segment_Mean"

## 1. cytoband matching

n <- nrow(blca_tcga); n;
blca_tcga$cyto <- NA; sum(is.na(blca_tcga$cyto))

for(i in 1:n){
	temp.i <- cyto[cyto$Chromosome == blca_tcga$Chromosome[i],]
	where <- (temp.i$start <= blca_tcga$Start[i] & temp.i$end >=blca_tcga$End[i])
  
 	if(sum(where) == 0) {print(paste0('cyto NA index :', i))}
 	else {blca_tcga$cyto[i] <- temp.i$name[where]}
}

colSums(is.na(blca_tcga)) # 79611 중 41541 na
cleanblca_tcga<-blca_tcga[complete.cases(blca_tcga$cyto), ] # dim : 38070

## na 처리

testna<-blca_tcga[!complete.cases(blca_tcga$cyto), ]
result_df <- data.frame();

for (i in 1:nrow(testna)) {
  start <- testna$Start[i]
  end <- testna$End[i]
  
  subset_rows <- cyto[(cyto$Chromosome==testna$Chromosome[i]) & (cyto$end >= start) & (cyto$start <= end), c("start","end","name")]
  nrows<-nrow(subset_rows)
  if (nrows == 0) {
    subset_rows <- data.frame(names = NA, start = start, end = end)
  } 
  
  subset_rows$x <- i
  subset_rows$Sample <- testna$Sample[i]
  subset_rows$Chrom <- testna$Chrom[i]
  subset_rows$Chromosome <- testna$Chromosome[i]
  subset_rows$Num_Probes <- testna$Num_Probes[i]
  subset_rows$Segment_Mean <- testna$Segment_Mean[i]	
  subset_rows[1,"start"] <-start
  subset_rows[nrows,"end"] <-end

  result_df <- rbind(result_df, subset_rows)
  print(i)

}

names(result_df)[1:3] <- c("Start", "End","cyto")
result_df <- subset(result_df, select = -x)
write.csv(result_df,"",, row.names = FALSE)

merged_df <- merge(result_df, cleanblca_tcga, by = c("Sample", "Chromosome", "Start", "End", "Num_Probes", "Segment_Mean", "cyto"), all = TRUE)
write.csv(merged_df,"C:/Users/judy0/OneDrive/바탕 화면/urep/0306mergedversion.csv",, row.names = FALSE)

## spread dataframe
cleanblca_tcga2<-merged_df%>% group_by(cyto,Sample) %>% summarise(m = mean(Segment_Mean,na.rm=T)) %>% spread(cyto,m)
write.csv(cleanblca_tcga2,"C:/Users/judy0/OneDrive/바탕 화면/urep/0306everythingfinish.csv",, row.names = FALSE)
