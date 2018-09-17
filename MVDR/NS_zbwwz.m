function lamda_d=NS_lx(abs_Y2,frame_number)
%h = waitbar(0,'Please wait...NS_lx ');
%噪声估计算法
lamda_d=zeros(706,frame_number);
%参数设置：
alpha_d=0.95;
alpha_s=0.2;
L=95;
alpha=0.8;
B=0.2;
c=1.25;
%segma(1:16)=3.6;segma(17:145)=2.5;segma(146:706)=1.2;
segma(1:265)=2;segma(266:441)=10;segma(442:706)=2;
S(:,1)=abs_Y2(:,1);
S_min(:,1)=abs_Y2(:,1);
P=zeros(706,frame_number);
%P(:,1)=zeros(:,1)
for k=1:1:706
lamda_d(k,1)=mean(abs_Y2(k,1:20));%第一帧噪声
lamda_dk(k,1)=mean(abs_Y2(k,1:20));%第一帧噪声
lamda_d(k,2)=mean(abs_Y2(k,1:20));%第一帧噪声
lamda_dk(k,2)=mean(abs_Y2(k,1:20));%第一帧噪声
end
%----------------------------------------
%对每帧进行噪声估计
for l=2:1:frame_number-1
    for k=1:706%对每个频点处理
        S(k,l)=alpha*S(k,l-1)+(1-alpha)*abs_Y2(k,l);
    end
end
for l=2:1:frame_number-1
    for k=1:706%对每个频点处理
       if l<L
           S_min1=min(S(k,1:l));
       else
           S_min1=min(S(k,l-L+1:l));
       end
       if l>frame_number-1-L-1
           S_min2=min(S(k,l:frame_number-1));
       else
           S_min2=min(S(k,l:l+L-1));
       end
  
             S_min(k,l)=max(S_min1,S_min2);
        %-----------------------------------------
        %语音存在概率的计算 
        if S_min(k,l)/S_min(k,l-1)<c&S_min(k,l)/S_min(k,l-1)>1/c
            lamda_dk(k,l+1)=S_min(k,l);
        else
            S_r(k,l)=S(k,l)/S_min(k,l);
            if S_r(k,l)>segma(k)
                I(k,l)=1;
            else
                I(k,l)=0;
            end
            P(k,l)=alpha_s*P(k,l-1)+(1-alpha_s)*I(k,l);
            %估计下一帧的噪声
            alpha_d_pie(k,l)=alpha_d+(1-alpha_d)*P(k,l);
            lamda_dk(k,l+1)=alpha_d_pie(k,l)*lamda_dk(k,l)+(1-alpha_d_pie(k,l))*abs_Y2(k,l);
        end
         lamda_d(k,l+1)= lamda_dk(k,l+1)/B;
         if lamda_d(k,l+1)==0;
             lamda_d(k,l+1)=0.001;
         end 
    end
  %waitbar(l/frame_number,h,num2str(fix(100*l/frame_number)))
end