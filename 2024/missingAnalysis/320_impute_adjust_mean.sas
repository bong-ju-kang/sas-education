/* 범주형 자료 기반으로 조정 셀 생성 */

data original_data;set hsb_mar; run;

/* Step 1: 각 조정 셀 내에서 결측값이 아닌 값의 평균을 계산 */
proc means data=original_data noprint;
    class female race; /* 성별과 연령대별로 그룹화 */
    var read; /* 대체할 변수 */
    output out=cell_means mean=cell_mean;
run;


/* Step 2: 원본 데이터와 평균값을 병합한 후, 결측값을 대체 */

/* 평균값을 저장한 데이터셋을 정렬  */
proc sort data=cell_means;
    by female race;
run;

proc sort data=original_data;
    by female race;
run;

data imputed_data;
    merge original_data(in=a) cell_means(in=b); /* 원본 데이터와 셀 평균 병합 */
    by female race; /* 성별 및 연령대 기준 병합 */
   
   	if a;
    
    /* 결측값을 셀 평균으로 대체 (original_data에 존재하고, read가  결측일 때만) */
    if missing(read) then read = cell_mean; /* 결측값을 해당 셀의 평균값으로 대체 */
run;
