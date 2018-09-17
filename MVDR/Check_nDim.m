function FoundNValBelowLim=Check_nDim(a,b,d,l,m,X_rcv,X_src,Rr,cc,MaxDelay,beta)
global SpectrumVec FreqPoints
FoundNValBelowLim=0;

n=1; %Cheek delay values for n=1 and above
dist=norm([2*a-1;2*b-1;2*d-1].*X_rcv+X_src+Rr.*[n;l;m]);%改过??????
foo_time=dist/cc;%计算镜像到麦克风的最大时延
while foo_time<=MaxDelay %if delay is below TF length limit for n=l，eheck n=2，3，4…
    foo_amplitude=prod(beta.^abs([n-a;n;l-b;l;m-d;m]))/(4*pi*dist);
    SpectrumVec=SpectrumVec+foo_amplitude*exp(i*2*pi*foo_time*FreqPoints);
    n=n+1;
    dist=norm([2*a-1;2*b-1;2*d-1].*X_rcv+X_src+Rr.*[n;l;m]);
    foo_time=dist/cc;
end
if n~=1
   FoundNValBelowLim=1;
end

n=0; %Cheek delay values for n=0 and below
dist=norm([2*a-1;2*b-1;2*d-1].*X_rcv+X_src+Rr.*[n;l;m]);
foo_time=dist/cc;
while foo_time<=MaxDelay %if delay is below TF length limit for n=l，eheck n=2，3，4…
    foo_amplitude=prod(beta.^abs([n-a;n;l-b;l;m-d;m]))/(4*pi*dist);
    SpectrumVec=SpectrumVec+foo_amplitude*exp(i*2*pi*foo_time*FreqPoints);
    n=n-1;
    dist=norm([2*a-1;2*b-1;2*d-1].*X_rcv+X_src+Rr.*[n;l;m]);
    foo_time=dist/cc;
end
if n~=0
   FoundNValBelowLim=1;
end

    

