library(dplyr);library(readr);library(tidyr)

normal_tcga <- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/CNV_Classifier2021/BLCA/BLCA_normal.txt",header=TRUE)
cyto <- read.table("C:/Users/judy0/OneDrive/바탕 화면/urep/0222_hg19_cytobandmatching/cytoBand_hg19.txt", sep = '\t');
colnames(cyto) <- c("chr","start","end","cyto","gieStain");
cyto$Chromosome <- substr(cyto$chr, 4, 10000);
cyto$name <- paste0("cyto.",cyto$Chromosome, cyto$cyto);
p <- length(cyto$name)

# colnames :  "Sample" "Chromosome" "Start" "End" "Num_Probes" "Segment_Mean"

# 1. cytoband matching

n <- nrow(normal_tcga); n;
normal_tcga$cyto <- NA; sum(is.na(normal_tcga$cyto))

for(i in 1:n){
	temp.i <- cyto[cyto$Chromosome == normal_tcga$Chromosome[i],]
	where <- (temp.i$start <= normal_tcga$Start[i] & temp.i$end >=normal_tcga$End[i])
  
 	if(sum(where) == 0) {print(paste0('cyto NA index :', i))}
 	else {normal_tcga$cyto[i] <- temp.i$name[where]}
}

colSums(is.na(normal_tcga)) # 24738 중 16315 na
cleannormal_tcga<-normal_tcga[complete.cases(normal_tcga$cyto), ]


## 2. na 처리

testna<-normal_tcga[!complete.cases(normal_tcga$cyto), ];
result_df <- data.frame();

# normal_tcga의 각 행에 대해 처리
for (i in 1:nrow(testna)) {
  start <- testna$Start[i]
  end <- testna$End[i]
  
  subset_rows <- cyto[(cyto$Chromosome==testna$Chromosome[i]) & (cyto$end >= start) & (cyto$start <= end), c("start","end","name")]
  nrows<-nrow(subset_rows)
  if (nrows == 0) {
    subset_rows <- data.frame(name = NA, start = start, end = end)
  } 
  
  subset_rows$x <- i
  subset_rows$Sample <- testna$Sample[i]
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

write.csv(result_df,"C:/Users/judy0/OneDrive/바탕 화면/urep/dirtytoclean_normal_중간단계.csv",, row.names = FALSE)

merged_df <- merge(result_df, cleannormal_tcga, by = c("Sample", "Chromosome", "Start", "End", "Num_Probes", "Segment_Mean", "cyto"), all = TRUE)
write.csv(merged_df,"C:/Users/judy0/OneDrive/바탕 화면/urep/0307mergedversion_normal.csv",, row.names = FALSE)

## 3. spread dataframe

cleannormal_tcga2<-merged_df%>% group_by(cyto,Sample) %>% summarise(m = mean(Segment_Mean,na.rm=T)) %>% spread(cyto,m)
write.csv(cleannormal_tcga2,"C:/Users/judy0/OneDrive/바탕 화면/urep/0307normaldone.csv",, row.names = FALSE)

### (참고) na 확인
non_zero_na_cols <- colnames(cleannormal_tcga2)[colSums(is.na(cleannormal_tcga2)) > 0];
na_counts <- colSums(is.na(cleannormal_tcga2));

for (col_name in non_zero_na_cols) {
  cat("열 이름:", col_name, "\tNA 개수:", na_counts[col_name], "\n")
}
