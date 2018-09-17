function [ h ] = ATF(Fs,X_rcv,X_src )
%ATF 此处显示有关此函数的摘要
%   此处显示详细说明
hn=hanning(706);
frame_number=1253;
j=sqrt(-1);
l=sqrt((X_rcv(1)-X_src(1))^2+(X_rcv(2)-X_src(2))^2+(X_rcv(3)-X_src(3))^2);
cita=1/(sqrt(4*pi)*l);
tao=(l/340)*Fs;

    for i=1:1:706
         v=(i-1)*Fs/706;
         h(i)=(1/l)*exp(-2*pi*j*l*v/340);
        
    end
    %h1=flip(h(2:353));
    %h=repmat(h,1,2);
    %h=[h,h1];
    h=h'; 
    h=repmat(h,1,frame_number);
end

