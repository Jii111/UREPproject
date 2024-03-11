# UREPproject
#### 이화여자대학교 통계학과 이동환 교수님 밑에서 진행한 UREP(학부생 연구인턴 프로그램) 프로젝트에 대한 페이지입니다.
#### 주제 : CNV 데이터 분석 및 방광암 분류 예측

<hr>

### 프로젝트 세부사항
- **사용 데이터**  
  : TCGA CNV blca & normal 데이터(snp_6 _Level_3__segmented_scna_minus_germline_cnv_hg19), 소변 데이터 BLCA_100k & NL_100k
  - **train & validation dataset** : TCGA dataset  
    -방광암(n=409) 79611 6  
    -정상(n=388) 24738 6  
  - **test dataset** : 소변 데이터  
    -방광암(BC, n=42) 166909 45  
    -정상(NL, n=28) 166909 31  
  
- **데이터 전처리**  
  **① Cytoband Matching**  
  **② NA 처리**  
  -데이터의 start, end가 같은 cytoband에 속해있지 않아서 na 발생  
    → cytoband(hg19)에 맞게 전처리  
  -이외 NA는 다변량 대치 방법 이용
      
  **⇒ 최종 데이터**  
  -UCSC에서 제공하는 염색체 구조에 따라 염색체를 약 800개 구역으로 분류
  -row 1개는 환자 1명의 case 의미
  -TCGA 방광암 : 409 787
  -TCGA 정상 : 388 788
  -소변 데이터 방광암 : 42 766
  -소변 데이터 정상 : 28 766

- **EDA** 

- **Modeling** 

<hr>

### 참고문헌
[1] Sohyun Im, et al. "Development of cancer classifier based on copy number variation in urinary cell-free DNA".  
[2] miniii222. (2019). CNV. https://github.com/miniii222/CNV  
[3] 홍다혜. (2019). Classification of cancer types based on DNA copy number variation (석사 학위 논문).   

