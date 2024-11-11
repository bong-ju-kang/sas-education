
data copy_hsb_mar;set hsb_mar; run;

/* Step 1: 범주형 변수의 빈도 계산 */
proc freq data=copy_hsb_mar noprint;
    tables prog / out=freq_table;
run;

/* Step 2: 누적 확률을 계산 (결측값은 제외) */
data freq_table_with_prob;
    set freq_table;
    if missing(prog) then delete; /* 결측값 제외 */
    
    retain cum_prob 0; /* 누적 확률 계산을 위한 변수를 유지 */
    prob = percent / 100; /* 각 범주의 확률 계산 (PROC FREQ에서 얻은 percent) */
    cum_prob + prob; /* 누적 확률 계산 */
run;


/* Step 3: 결측값을 확률적으로 대체 */
data imputed_data;
    set copy_hsb_mar;
    
    if missing(female) then do;
        rand_num = ranuni(0); /* 0과 1 사이의 랜덤 숫자를 생성 */
        
        /* 누적 확률을 기반으로 범주 대체 */
		if rand_num <= 0.522 then prog = 'academic'; 
       	else if rand_num <= 0.7472527473 then prog='general';
        else prog = 'vocation'; 
    end;
run;


/* Step 4: 결과 확인 */
proc freq data=copy_hsb_mar;
    tables prog ;
run;


proc freq data=imputed_data;
    tables prog ;
run;
