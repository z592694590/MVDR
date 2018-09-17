function [lamda_d]=NE(Y,frame_number)

abs_Y=abs(Y);

abs_Y2=abs_Y.^2;
lamda_d=NS_zbwwz(abs_Y2,frame_number);%ÔëÉù¹À¼Æ