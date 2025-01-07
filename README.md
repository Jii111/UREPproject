# UREPproject
#### 이화여자대학교 통계학과 UREP(학부생 연구인턴 프로그램) 프로젝트에 대한 페이지입니다.
#### 주제 : CNV 데이터 분석 및 암 분류 예측
   
   
### 📑 프로젝트 세부사항
- 2024.01 ~ 2024.05  
- **사용 데이터**  
  : 소변 데이터 BLCA_15k & NL_15k & PRAD_15k & RCC_15k  
  - train/test 데이터로 분리 후 train data에 GAN 적용  
    *GAN을 사용하지 않고 TCGA CNV 데이터(snp_6 _Level_3__segmented_scna_minus_germline_cnv_hg19)를 사용할 수 있으나, 두 데이터간 이질성이 존재하는 것으로 확인
    ---------------------|-----------------|---------------------|---------------
     방광암(BC, n=42)    | (166909, 45)    | 전립선암(PRAD, n=26) | (166909, 26)    
     정상(NL, n=28)      | (166909, 31)    | 신장암(RCC, n=388)   | (166909, 26)     
   
    
- **데이터 전처리**  
  **① Cytoband Matching**  
  · row 1개에 환자 1명의 정보를 담기 위해 format 변환  
  · UCSC에서 제공하는 염색체 구조에 따라 염색체를 약 800개 구역으로 분류  
  · 이 정보와 데이터의 염기서열 정보를 통해 해당되는 cytoband 매칭  
      
  **⇒ 최종 데이터**  
    · 방광암(BC, n=42) 42 769   
    · 정상(NL, n=28) 28 769  
    · 전립선암(PRAD, n=26) 23 769  
    · 신장암(RCC, n=388) 23 769  
    
  **② 데이터 증강(GAN)**  
  · 적은 데이터의 수로 인해 딥러닝 학습이 어렵다고 판단하여 데이터 증강  
  · GAN / CGAN / CTGAN 모델 적용 후 예측 성능 비교  
    
- **EDA**    
  · 기본 정보 확인(shape,null,info)  
  · y 분포 확인 / column별 비교 / case에 따른 염색체 영역 정보 변화 추이  
  · 염색체 영역(column) 간 상관성  
   
- **Modeling: TLTD**   
  · tabular 데이터의 딥러닝 학습 성능을 향상시키기 위해 데이터를 이미지 형태로 변환하고, distillation technique를 사용하는 방법  
  ① 이미지 변환 : Tab2Img  
  ② Teacher model : pre-trained ResNet50  
  ③ Student model : FCNN  
  ④ 최종 분류 모델 : Random Forest  
    
- **Result**    
  · GAN 모델 별/예측 모델별 성능 비교  
  · 기본 GAN 모델로 4배를 증강시킨 데이터셋에서 모델 성능이 가장 뛰어난 것을 확인  
  · SHAP summary plot / ROC curve 결과 확인  
    
  · 소변 세포유리 DNA 데이터로 암을 효과적으로 예측할 수 있음 확인  
  · GAN 모델로 데이터를 증강시키고 TLTD 방법을 적용하여, 데이터 크기의 부족이라는 한계를 극복하고 예측 모델 성능을 개선하는 방법을 제안  
  · 데이터를 이미지로 변환하여 distillation techinique을 적용한 TLTD 방법이 다른 전통적인 머신 러닝 방법에 비해서도 뛰어난 성능을 보이는 것을 확인  
  · 테스트 데이터 또한 사이즈가 크지 않다는 점을 고려하였을 때, 추가적인 검증 필요  
    
*2024 한국보건정보통계학회 추계학술대회에 포스터 논문을 작성하였고, 우수 연구 발표 후 우수 구연상을 수상하였습니다.  
    
   
### 📑 결과 정리
- **초기 결과물** : https://jiiiiiii11.notion.site/8099fbc463dc4b86bbb5df115aa58f98?pvs=4
- **한국보건정보통계학회 포스터 논문** : https://drive.google.com/file/d/19KhQyWfiZpTxb-7a2gL7CQXi0N3LEfnV/view?usp=sharing
- **한국보건정보통계학회 발표 자료** : https://drive.google.com/file/d/1T50OJpyBIstESMdEgqetcS-gAUMifA_c/view?usp=sharing

#### 📑 참고문헌
[1] Sohyun Im, et al. "Development of cancer classifier based on copy number variation in urinary cell-free DNA".  
[2] miniii222. (2019). CNV. https://github.com/miniii222/CNV  
[3] 홍다혜. (2019). Classification of cancer types based on DNA copy number variation (석사 학위 논문).   

