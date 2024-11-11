/* 회귀 대체 */
data copy_hsb_mar;set hsb_mar;run;

/* Step 1: 범주형 변수 'prog'를 더미 변수로 변환 */
proc glmselect data=copy_hsb_mar;
    class prog; /* 범주형 변수 선언 */
    model read = prog write math science / selection=none;
    output out=regression_out p=pred_read; /* 예측값을 새로운 변수로 저장 */
run;

/* Step 2: 결측값 대체 - 예측된 값으로 'read'의 결측값 대체 */
data hsb_mar_imputed;
    set regression_out;
    if missing(read) then read = pred_read; /* read가 결측인 경우 예측값으로 대체 */
run;

/* Step 3: 결과 확인 */
proc print data=hsb_mar_imputed;
    var ID read write math science prog;
run;

/* 공분산 비교 */
proc corr data=hsb2 cov;
	var read write math science;
run;

proc corr data=hsb_mar_imputed cov;
	var read write math science;
run;