/* 회귀 대체 */
data copy_hsb_mar;set hsb_mar;run;

/* glm 적합 */
proc glmselect data=copy_hsb_mar;
    class prog; /* 범주형 변수 선언 */
    model read = prog write math science / selection=none;
    output out=regression_out p=p_read r=r_read; 
run;

/* SSE 계산  */
proc sql noprint;
    select sum(r_read**2) into :SSE
    from regression_out
    where not missing(r_read);
quit;
%PUT &SSE;

/* 유효 관측치 수(n) 계산 */
proc sql noprint;
    select count(*) into :n
    from regression_out
    where not missing(r_read);
quit;
%PUT &n;

/* 모델의 p 계산: 설명 변수의 수 + 1 (절편) */
%let p = 4 + 1; /* PROG, WRITE, MATH, SCIENCE + 절편 */

/* 잔차 분산 추정치 계산: SSE / (n - p) */
%let variance = %sysevalf(&SSE / (&n - &p));
%put &variance;



/* 확률적 회귀 대체 수행 */
data hsb_mar_imputed;
    set regression_out;
    if missing(READ) then do;
        epsilon = rannor(12345) * sqrt(&variance); /* 우변의 의미: N(0, variance) 에서 추출과 동일 */
        READ = p_read + epsilon;
    end;
run;


/* 결과 확인 */
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