/* 가변수 생성 */
data hsb_mar_dummy;
	set hsb_mar;
	if prog ^= " " then do;
		if prog ="academic" then prog_academic=1;
		else prog_academic=0;
		if prog ="general" then prog_general=1;
		else prog_general=0;
	end;
	if female ^= " " then do;
		if female="female" then female_female = 1;
		else female_female = 0;
	end;
run;