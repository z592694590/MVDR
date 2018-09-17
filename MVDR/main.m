%麦克风阵列信号产生
clear all
close all
clc
Fs=8000;
beta=[0 0 0 0 0 0];%反射系数 文章P24
source=[20 25 3]; %声源坐标
noise=[3.5 2.5 1];%噪声坐标
sink=0+50*rand(10,2);
sink(:,3)=1;
sink(2,:)=[35 25 1];
sink_1=[2 23.9 3];% 麦克风坐标
sink_2=[1 3.4 1];
sink_3=[2 3.5 1.9];
sink_4=[1 3.4 1];
sink=[sink_1;sink_2;sink_3;sink_4]

sink_5=[1 3.1 1];
sink_6=[0.5 2.5 1];
sink_7=[2 3 1];
sink_8=[0.1 2.5 1];
sink_9=[2 0.5 1];
sink_10=[0.4 1.5 1];

room=[50 40 30];%房间大小
%LimAtten=30;%脉冲响应衰减阈值
%measT60=0;%混响时间    
%snr=20;%信噪比
%c=345;
%计算各麦克风处房间脉冲响应
%h_1=MakeIMResp(FS,beta,sink_1,source,room,c,LimAtten,measT60);
%h_2=MakeIMResp(Fs,beta,sink_2,source,room,c,LimAtten,measT60);
%h_3=MakeIMResp(Fs,beta,sink_3,source,room,c,LimAtten,measT60);
%h_4=MakeIMResp(Fs,beta,sink_4,source,room,c,LimAtten,measT60);


[s1,FS]=audioread('西藏大嘴乌鸦纯净.wav');%原始信号
%[n,wn,beta]=kaiserord([0.065,0.085]*pi,[1,0],[0.01,0.01],2*pi);%滤除300HZ以下噪声
%hh=fir1(n,wn,'high',kaiser(n+1,beta),'noscale');
n=mvnrnd (0,0.003,442662 ) ;%产生白噪声
n=[mvnrnd(0,0.0001,100000);mvnrnd(0,0.002,121331);mvnrnd(0,0.003,221331)] ;
n=[mvnrnd(0,0.001,100000);mvnrnd(0,0.02,121331);mvnrnd(0,0.05,221331)] ;
n1=mvnrnd (0,0.000001,442662 ) ;
n2=mvnrnd (0,0.02,442662 ) ; ;
n3=mvnrnd (0,0.004,442662 ) ; ;
n4=mvnrnd (0,0.01,442662 ) ; ;

for i=1:length(sink)
    n(:,i*442662)=[mvnrnd(0,0.001,100000);mvnrnd(0,0.02,121331);mvnrnd(0,0.005,221331)]';
end    
n5=randn(1,100)

n1=[mvnrnd(0,0.0001,100000 );mvnrnd(0,0.0002,221331);mvnrnd(0,0.0003,121331)] ; 
n2=[mvnrnd(0,0.001,100000 );mvnrnd(0,0.02,221331);mvnrnd(0,0.2,121331)] ; 
n3=[mvnrnd(0,0.02,100000 );mvnrnd(0,0.04,221331);mvnrnd(0,0.3,121331)] ; 
n4=[mvnrnd(0,0.01,100000 );mvnrnd(0,0.2,221331);mvnrnd(0,0.8,121331)] ; 

plot(s1,'k')
figure
plot(n,'k')
xlabel('Sampling point')
ylabel('Normalized amplitude')

n=wgn(1,442662,0);
%s=filter(hh,1,s1(:,1));
s=s1(:,1);
%麦克风处信号
%语音信号预处理
hn=hanning(706);										
d_frame=(enframe(s,706,353))';%分帧
size_frame=size(d_frame);
frame_number=size_frame(2);%帧数
for l=1:1:frame_number
    Y(:,l)=fft(d_frame(:,l).*hn,706);%添加分析汉宁窗   
    
end

d_frame=(enframe(n,706,353))';%分帧
size_frame=size(d_frame);
frame_number=size_frame(2);%帧数
for l=1:1:frame_number
    N(:,l)=fft(d_frame(:,l).*hn,706);%添加分析汉宁窗   

end

hn=hanning(706);
for i=1:length(sink)
    n=[mvnrnd(0,0.0001,100000);mvnrnd(0,0.002,121331);mvnrnd(0,0.003,221331)] ;
    d_frame=(enframe(n,706,353))';%分帧
    size_frame=size(d_frame);
    frame_number=size_frame(2);%帧数
    for l=1:1:frame_number
        N(:,l+(i-1)*frame_number)=fft(d_frame(:,l).*hn,706);%添加分析汉宁窗   

    end
end
df=FS/(length(N)-1);
f=(0:length(N)-1)*df;
plot(f(1:length(N)),abs(Q(1:length(N))));
 
%计算声传函数
h=[];n1=[];
for i=1:length(sink)
    h(:,(i-1)*frame_number+1:i*frame_number)=ATF(FS,sink(i,:),source);
    n1(:,(i-1)*frame_number+1:i*frame_number)=ATF(FS,sink(i,:),noise);
end
for i=1:20
  
    n1(:,(i-1)*frame_number+1:i*frame_number)=ATF(FS,sink(i,:),noise);
end

for i=1:length(sink)
    h(i)=ATF1(FS,sink(i,:),source);
    n1(i)=ATF1(FS,sink(i,:),noise);
end  
h_1=ATF1(FS,sink_1,source);
h_2=ATF(FS,sink_2,source);
h_3=ATF(FS,sink_3,source);
h_4=ATF(FS,sink_4,source);
h_5=ATF(FS,sink_5,source);
h_6=ATF(FS,sink_6,source);
h_7=ATF(FS,sink_7,source);
h_8=ATF(FS,sink_8,source);
h_9=ATF(FS,sink_9,source);
h_10=ATF(FS,sink_10,source);
n_1=ATF1(FS,noise,sink_1);
n_2=ATF(FS,noise,sink_2);
n_3=ATF(FS,noise,sink_3);
n_4=ATF(FS,noise,sink_4);
n_5=ATF(FS,noise,sink_5);
n_6=ATF(FS,noise,sink_6);
n_7=ATF(FS,noise,sink_7);
n_8=ATF(FS,noise,sink_8);
n_9=ATF(FS,noise,sink_9);
n_10=ATF(FS,noise,sink_10);

%x1=conv(h_1,s);
%x2=conv(h_2,s);
%x3=conv(h_3,s);
%x4=conv(h_4,s);
%s=fft(s);
%s2=real(ifft(s))
%h1=ifft(h_1);h2=ifft(h_2);h3=ifft(h_3);h4=ifft(h_4);
%x1=conv(h1,s)+n;x2=conv(h2,s)+n;x3=conv(h3,s)+n;x4=conv(h4,s)+n;

frame_number=1253;
x3=x(:,3*1253+1:4*1253);
for l=1:1:frame_number%ifft
    X0(:,l)=real(ifft(x3(:,l),706));
end
X1=zeros(frame_number*353+353,1);
for l=1:1:frame_number%分帧叠加
    X2((l-1)*353+1:(l+1)*353)=X1((l-1)*353+1:(l+1)*353)+X0(:,l);
end

plot(MSEi);
hold on;
plot(SNRi);


x=zeros(706,20*frame_number);
for i=1:length(sink)
    x(:,(i-1)*frame_number+1:i*frame_number)=h(:,(i-1)*frame_number+1:i*frame_number).*Y+n1(:,(i-1)*frame_number+1:i*frame_number).*N;
end
for i=1:length(sink)
    x(:,(i-1)*frame_number+1:i*frame_number)=h(:,(i-1)*frame_number+1:i*frame_number).*Y+N(:,(i-1)*frame_number+1:i*frame_number);
end
for i=1:length(sink)
    x(:,(i-1)*frame_number+1:i*frame_number)=h(i).*Y+n1(i)*N;
end

%计算每个节点的MSE和SNR
for i=1:length(sink)
    M=abs(x(:,(i-1)*frame_number+1:i*frame_number)-Y).^2;
    MSEi(i)=10*log10(sum(M(:))/(706*1253));
end

for i=1:length(sink)
    M1=sum(((abs(Y).^2)./(abs(x(:,(i-1)*frame_number+1:i*frame_number)-Y).^2)),1);
    SNRi(i)=1/1256*(sum(10*log10(M1),2));
end

  Z=Y+N1;
  M=sum(abs(Y).^2,1)./sum(abs(Z-Y).^2,1);
  SNR=1/1256*sum(10*log10(M),2);

M=abs(x3-Y).^2;
MSE=10*log10(sum(M(:))/(706*1253));

x1=h(:,0*frame_number+1:1*frame_number).*Y;
%x1=real(ifft(x1));
x2=h_2.*Y+N2;
%x2=real(ifft(x2));
x3=h_3.*Y+N3;
%x3=real(ifft(x3));
x4=h_4.*Y+N4;
%x4=real(ifft(x4));

M=abs(x3-Y).^2;
MSE1=10*log10(sum(M(:))/(706*1253));

x5=h_5*s+n_5*n;
x5=real(ifft(x5));
x6=h_
x10=real(ifft(x10));6*s+n_6*n;
x6=real(ifft(x6));
x7=h_7*s+n_7*n;
x7=real(ifft(x7));
x8=h_8*s+n_8*n;
x8=real(ifft(x8));
x9=h_9*s+n_9*n;
x9=real(ifft(x9));
x10=h_10*s+n_10*n;

%x_1=awgn(x1,snr,'measured');%加入SNR为零的正态分布白噪声随机序列 （目前不需要）
%x_2=awgn(x2,snr,'measured');
%x_3=awgn(x3,snr,'measured');
%x_4=awgn(x4,snr,'measured');

%x0=[2 1 2 3 2];%三维图参数
%y1=[3.5 1.84 1 2 2.5];
%z1=[2 1 1 1 1];

x0=sink(:,1)';
x0=[x0(1:0) source(1) x0(1:end)];
y1=sink(:,2)';
y1=[y1(1:0) source(2) y1(1:end)];
z1=sink(:,3)';
z1=[z1(1:0) 5 z1(1:end)];
c0(1) = 1;
c0(2:length(sink)+1) = 7;
s0(1:length(sink)+1) = 100;
figure;
scatter3(x0,y1,z1,s0,c0,'fills');
xlabel('x(m)');ylabel('y(m)');zlabel('z(m)');
axis([0 50 0 50 1 5]);
text(x0(1)+2,y1(1),z1(1),'source');
for i=2:length(sink)+1;
    text(x0(i)+1.3,y1(i),z1(i),num2str(i-1));
end
figure;
scatter3(x0,y1,z1,s0,c0,'fills');
axis([0 50 0 50]);
xlabel('x(m)');ylabel('y(m)');zlabel('z(m)');
view([0 90]);
L=zeros(length(sink),length(sink));

text(x0(1)+1.3,y1(1),z1(1),'source');
for i=2:length(sink)+1;
    text(x0(i)+1,y1(i),z1(i),num2str(i-1));
end

for i=2:1:length(sink)+1 %生成拓扑结构
    for j=i+1:1:length(sink)+1
        if sqrt((x0(i)-x0(j))^2+(y1(i)-y1(j))^2 )<20 %通信半径为4m
            line([x0(i),x0(j)],[y1(i),y1(j)])
            L(i-1,j-1)=-1;%拉普拉斯连接矩阵
            L(j-1,i-1)=-1;
        end
    end
end
%%+
%语音预处理，分帧加窗

%噪声谱估计
%4节点
for i=1:length(sink)
    [lamda_d(:,(i-1)*frame_number+1:i*frame_number)]=NoiseEstimation(x(:,(i-1)*frame_number+1:i*frame_number),frame_number);
end
[lamda_d1]=NoiseEstimation(x1,frame_number);
[lamda_d2]=NoiseEstimation(x2,frame_number);
[lamda_d3]=NoiseEstimation(x3,frame_number);
[lamda_d4]=NoiseEstimation(x4,frame_number);
[lamda_d5,x5]=NoiseEstimation(x5);
[lamda_d6,x6]=NoiseEstimation(x6);
[lamda_d7,x7]=NoiseEstimation(x7);
[lamda_d8,x8]=NoiseEstimation(x8);
[lamda_d9,x9]=NoiseEstimation(x9);
[lamda_d10,x10]=NoiseEstimation(x10);
%%
%%VAD估计
for i=1:length(sink)
    lamda_v(i)=vad(x(:,(i-1)*frame_number+1:i*frame_number));
end
[lamdav_1]=vad(x1);
[lamdav_2]=vad(x2);
[lamdav_3]=vad(x3);
[lamdav_4]=vad(x4);


%%
%%波束形成算法
Y_=[];
for i=1:length(sink)
    Y_(:,(i-1)*frame_number+1:i*frame_number)=conj(h(:,(i-1)*frame_number+1:i*frame_number)).*lamda_d(:,(i-1)*frame_number+1:i*frame_number).^-1.*x(:,(i-1)*frame_number+1:i*frame_number);
end

for i=1:length(sink)
    N_(:,(i-1)*frame_number+1:i*frame_number)=h(:,(i-1)*frame_number+1:i*frame_number).*lamda_d(:,(i-1)*frame_number+1:i*frame_number).^-1.*conj(h(:,(i-1)*frame_number+1:i*frame_number));
end

for i=1:length(sink)
    Y_(:,(i-1)*frame_number+1:i*frame_number)=h(i)'.*lamda_d(:,(i-1)*frame_number+1:i*frame_number).^-1.*x(:,(i-1)*frame_number+1:i*frame_number);
end
for i=1:length(sink)
    N_(:,(i-1)*frame_number+1:i*frame_number)=h(i).*lamda_d(:,(i-1)*frame_number+1:i*frame_number).^-1.*h(i)';
end

for i=1:length(sink)
    Y_v(:,(i-1)*frame_number+1:i*frame_number)=conj(h(:,(i-1)*frame_number+1:i*frame_number)).*lamda_v(i).^-1.*x(:,(i-1)*frame_number+1:i*frame_number);
end

for i=1:length(sink)
    N_v(:,(i-1)*frame_number+1:i*frame_number)=h(:,(i-1)*frame_number+1:i*frame_number).*lamda_v(i).^-1.*conj(h(:,(i-1)*frame_number+1:i*frame_number));
end


for i=1:length(sink)
    Y_v(:,(i-1)*frame_number+1:i*frame_number)=h(i)'*lamda_v(i).^-1.*x(:,(i-1)*frame_number+1:i*frame_number);
end

for i=1:length(sink)
    N_v(i)=h(i)*lamda_v(i).^-1.*h(i)';
end


%新算法
Y_sum=zeros(706,1253);
N_sum=zeros(706,1253);
for i=1:length(sink)
    Y_sum=Y_sum+Y_(:,(i-1)*frame_number+1:i*frame_number);
    N_sum=N_sum+N_(:,(i-1)*frame_number+1:i*frame_number);
end
Z=Y_sum./N_sum;

M=abs(Z-Y).^2;
MSEd=10*log10(sum(M(:))/(706*1253));

M2=sum((abs(Y).^2),1)./sum((abs(Z-Y).^2),1);
M2=sum(((abs(Y).^2)./(abs(Z-Y).^2)),1);
SNRd=1/1253*(sum(10*log10(M2),2));
%老算法
Y_sumv=zeros(706,1253);
N_sumv=zeros(706,1253);
for i=1:length(sink)
    Y_sumv=Y_sumv+Y_v(:,(i-1)*frame_number+1:i*frame_number);
    N_sumv=N_sumv+N_v(i);
end
Z_v=Y_sumv./N_sumv;

M=abs(Z_v-Y).^2;
MSEv=10*log10(sum(M(:))/(706*1253));

M3=sum((abs(Y).^2),1)./sum((abs(Z_v-Y).^2),1);
M3=sum(((abs(Y).^2)./(abs(Z_v-Y).^2)),1);
SNRv=1/1253*sum((10*log10(M3)),2);

Y_1=conj(h_1).*lamdav_1.^-1.*x1;
N_1=h_1*lamdav_1'.^-1.*conj(h_1);
Y_2=conj(h_2).*lamdav_2.^-1.*x2;
N_2=h_2*lamdav_2.^-1.*conj(h_2);
Y_3=conj(h_3).*lamdav_3.^-1.*x3;
N_3=h_3*lamdav_3.^-1.*conj(h_3);
Y_4=conj(h_4).*lamdav_4.^-1.*x4;
N_4=h_4*lamdav_4.^-1.*conj(h_4);

Y_1=conj(h_1).*lamda_d1.^-1.*x1;
N_1=h_1.*lamda_d1.^-1.*conj(h_1);
Y_2=conj(h_2).*lamda_d2.^-1.*x2;
N_2=h_2.*lamda_d2.^-1.*conj(h_2);
Y_3=conj(h_3).*lamda_d3.^-1.*x3;
N_3=h_3.*lamda_d3.^-1.*conj(h_3);
Y_4=conj(h_4).*lamda_d4.^-1.*x4;
N_4=h_4.*lamda_d4.^-1.*conj(h_4);


Y_1=h_1'*x1;
N_1=h_1*h_1';
Y_2=h_2'.*x2;
N_2=h_2*h_2';
Y_3=h_3'*x3;
N_3=h_3*h_3';
Y_4=h_4'*x4;
N_4=h_4*h_4';

Y_1=real(h_1')*lamda_d1.^-1.*x1;
N_1=real(h_1')*lamda_d1.^-1*real(h_1);
Y_2=real(h_2')*lamda_d2.^-1.*x2;
N_2=real(h_2')*lamda_d2.^-1*real(h_2);
Y_3=real(h_3')*lamda_d3.^-1.*x3;
N_3=real(h_3')*lamda_d3.^-1*real(h_3);
Y_4=real(h_4')*lamda_d4.^-1.*x4;
N_4=real(h_4')*lamda_d4.^-1*real(h_4);
Y_5=h_5'*lamda_d5.^-1.*x5;
N_5=h_5'*lamda_d5.^-1*h_5;
Y_6=h_6'*lamda_d6.^-1.*x6;
N_6=h_6'*lamda_d6.^-1*h_6;
Y_7=h_7'*lamda_d7.^-1.*x7;
N_7=h_7'*lamda_d7.^-1*h_7;
Y_8=h_8'*lamda_d8.^-1.*x8;
N_8=h_8'*lamda_d8.^-1*h_8;
Y_9=h_9'*lamda_d9.^-1.*x9;
N_9=h_9'*lamda_d9.^-1*h_9;
Y_10=h_10'*lamda_d10.^-1.*x10;
N_10=h_10'*lamda_d10.^-1*h_10;
Z=(Y_1+Y_2+Y_3+Y_4+Y_5+Y_6+Y_7+Y_8+Y_9+Y_10)./(N_1+N_2+N_3+N_4+N_5+N_6+N_7+N_8+N_9+N_10);
Z=(Y_1+Y_2+Y_3+Y_4)./(N_1+N_2+N_3+N_4);

%评价指标MSEframe_number=1253;

M=abs(Z-Y).^2;
MSE=10*log10(sum(M(:))/(706*1253));%分帧叠加
for l=1:1:frame_number%ifft
    X0(:,l)=real(ifft(Z_v(:,l),706));
end
X1=zeros(frame_number*353+353,1);
for l=1:1:frame_number%分帧叠加
    X1((l-1)*353+1:(l+1)*353)=X1((l-1)*353+1:(l+1)*353)+X0(:,l);
end



%%
plot(s);
xlabel('Sampling point')
ylabel('Amplitude')
%% 一致性算法平均Metropolis算法
%%
x2(221415:221469,:)=0;
x3(221463:221469,:)=0;
x4(221433:221469,:)=0;
%x = [x1 x2 x3 x4];
%x=[1 2 3 4]

%Y_ave=[reshape(Y_1,1,[]);reshape(Y_2,1,[]);reshape(Y_3,1,[]);reshape(Y_4,1,[]);]
%N_ave=[reshape(N_1,1,[]);reshape(N_2,1,[]);reshape(N_3,1,[]);reshape(N_4,1,[]);]
for i=1:length(sink)
    Y_ave(i,:)=reshape(Y_(:,(i-1)*frame_number+1:i*frame_number),1,[]);
    N_ave(i,:)=reshape(N_(:,(i-1)*frame_number+1:i*frame_number),1,[]);
end    
x = x';
p = -L;
cita=0.95;
c = sum(p,2);
for i=1:length(sink)
    p(i,i)=c(i,1);
end    
%p = eye(4)-0.05*p;%一致性P矩阵
for i=1:1:length(sink)
    for j=i+1:length(sink)
        if p(i,j)~=0
            %p(i,j)=2*cita/(p(i,i)+p(j,j));%平均
            p(i,j)=1/max(p(i,i),p(j,j));%Metroplois
            %p(i,j)=1/length(sink);%最大度
            p(j,i)=p(i,j);
        end
    end
end
for i=1:length(sink)
    p(i,i)=0;
    p(i,i)=1-sum(p(i,:),2);
    %p(i,i)=1-c(i)/length(sink);%最大度
end


for i=1:50
   x=p*x;
   y(:,i)=x;
end
plot(y')
  Y_ave=p*Y_ave;
  N_ave=p*N_ave;
for j=1:100
    
        Y_ave=p*Y_ave;
        N_ave=p*N_ave;
        %y(:,i) = x(:,i); 
  
        Z=Y_ave(2,:)./N_ave(2,:);
        Z=reshape(Z,706,1253);
        M=abs(Z-Y).^2;
        MSE1(j)=10*log10(sum(M(:))/(706*1253));
end
MSE=[MSE(1:0) MSEi(2) MSE(1:end)];
MSE1=[MSE1(1:0) MSEi(2) MSE1(1:end)];
MSE2=[MSE2(1:0) MSEi(2) MSE2(1:end)];
l=0:1:100;
plot(l,MSE,'o-');
hold on;
plot(l,MSE1,'s-');
hold on;
plot(l,MSE2,'*-');
axis([0 25 -12 3]);
legend('Average Metropolis-weight','Metropolis-weight','Max Degree');
xlabel('number of fiexd iterations');
ylabel('MSE(db)');
%MSE=[MSE(1:0) 17.9732 MSE(1:end)];

Z=reshape(Z,706,1253);
M=abs(Z-Y).^2;
MSE2=10*log10(sum(M(:))/(706*1253));



figure;
subplot(411);
plot(y(1,:))
title('一致性更新结果')
subplot(412);
plot(y(2,:))
subplot(413);
plot(y(3,:))
subplot(414);
plot(y(4,:))
%%

figure;
subplot(511)
plot(real(ifft(s)))
title('原始信号')
subplot(512)
plot(real(x1))
title('麦克风1处得到的信号')
subplot(513)
plot(real(x2))
title('麦克风2处得到的信号')
subplot(514)
plot(real(x3))
title('麦克风3处得到的信号')
subplot(515)
plot(real(x4))
title('麦克风4处得到的信号')
audiowrite('out1.wav',X1,FS);audiowrite('out2.wav',X2,FS);
