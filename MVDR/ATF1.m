function [ h ] = ATF1(Fs,X_rcv,X_src )
%ATF 此处显示有关此函数的摘要
%   此处显示详细说明
F=Fs/1000;
frame_number=1253;

l=sqrt((X_rcv(1)-X_src(1))^2+(X_rcv(2)-X_src(2))^2+(X_rcv(3)-X_src(3))^2);
cita=1/l;
tao=(l/340);
w=2*pi;
j=sqrt(-1);
h=cita*exp(-2*j*w*tao);
%h=cita;  
end