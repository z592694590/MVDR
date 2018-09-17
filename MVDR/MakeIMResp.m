%2014/6/10
%混响声学房间模型的建立（IMAGE 模型  镜像模型）
%《阵列数字助听系统核心问题的研究》 孟君
function FilterCoeffs=MakeIMResp(Fs,beta,X_rcv,X_src,room,cc,LimAtten_dB,measT60) %自己加的  FilterCoeffs还没弄明白
global SpectrumVec FreqPoints
%*********Check user input
if X_rcv(1)>=room(1)| X_rcv(2)>=room(2)| X_rcv(3)>=room(3)| X_rcv(1)<=0|X_rcv(2)<=0|X_rcv(3)<=0,
    error('Receiver must be within the room boundaries');
elseif X_src(1)>=room(1)| X_src(2)>=room(2)| X_src(3)>=room(3)| X_src(1)<=0|X_src(2)<=0|X_src(3)<=0,
     error('source must be within the room boundaries');
elseif ~isempty(find(beta>=1))|~isempty(find(beta<0)),
   error('Parameter "beta" must be in range[0...1]');

end

if nargin<9 %输入参数的个数
    silentflag=0;  %set to 1 to disable on-screen messages
end

X_src=X_src(:);%Source location
X_rcv=X_rcv(:);%Receiver location
beta=beta(:);
Rr=2*room(:);  %Room dimensions
DPdel=norm(X_src-X_rcv)/cc; %direct path delay in [s]?????

%********Define enough frequency points for resulting time impulse response
if ~isequal(beta,zeros(6,1)),
    if isempty(measT60), %if no Practieal T60 measurement available，use Sabine estimate
          V=prod(room);
          aa_sub=(2-beta(1)^2-beta(2)^2)*room(2)*room(3)
                 +(2-beta(3)^2-beta(4)^2)*room(1)*room(3)
                 +(2-beta(5)^2-beta(6)^2)*room(1)*room(2);
          T60val=0.161*V/aa_sub;  %Sabine's reverberation time in [s]
    else
          T60val=measT60;  % Practical T60 measurement determines real energy decay in TF!
    end
    foo=LimAtten_dB*T60val/60; %desired length of TF(TF decays by 60dB for T60 seconds after direct Path delay)
    MaxDelay=DPdel+foo; % maximum delay in TF: direct path plus TForder
else
     MaxDelay=2*DPdel;  % anechoic case: allow for 2 times direct Path in TF
end
TForder=ceil(MaxDelay*Fs); %total TF length; ceil找离它最近的整数 在0~Fs/2范围内共有TForder个点

FreqPoints=linspace(0,Fs/2,TForder)';
SpectrumVec=zeros(TForder,1);

%--------summation over room dimensions
if ~silentflag 
    fprintf('[MakeIMResp]Computing transfer function');
end
for a=0:1
    for b=0:1
        for d=0:1
            if ~silentflag
                fprintf('.');
            end
            m=1; %Cheek delay values for m=1 and above
            FoundLValBelowLim=Check_lDim(a,b,d,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
            while FoundLValBelowLim==1
                m=m+1;
                FoundLValBelowLim=Check_lDim(a,b,d,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
            end
            
            m=0; %Cheek delay values for m=0 and below
            FoundLValBelowLim=Check_lDim(a,b,d,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
             while FoundLValBelowLim==1
                m=m-1;
                FoundLValBelowLim=Check_lDim(a,b,d,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
             end
            
        end
    end
end
if ~silentflag
    fprintf('\n');
end

%------Inverse Fourier transform
SpectrumVec(1)=SpectrumVec(1)/2; % remove DC component in resulting time coefficients
freqvec=-i*2*pi*linspace(0,1/2,TForder);
FilterCoeffs=zeros(TForder,1);
for ii=1:TForder
    freq=exp((ii-1)*freqvec);
    FilterCoeffs(ii)=real(freq*SpectrumVec);%?????????
end
 FilterCoeffs= FilterCoeffs/TForder;

     



