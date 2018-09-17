function FoundLValBelowLim=Check_lDim(a,b,d,m,X_rcv,X_src,Rr,cc,MaxDelay,beta)

FoundLValBelowLim=0;

l=1; %Cheek delay values for l=1 and above
FoundNValBelowLim=Check_nDim(a,b,d,l,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
while FoundNValBelowLim==1
    l=l+1;
    FoundNValBelowLim=Check_nDim(a,b,d,l,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
end
if l~=1
   FoundLValBelowLim=1;
end

l=0; %Cheek delay values for l=0 and below
FoundNValBelowLim=Check_nDim(a,b,d,l,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
while FoundNValBelowLim==1
    l=l-1;
    FoundNValBelowLim=Check_nDim(a,b,d,l,m,X_rcv,X_src,Rr,cc,MaxDelay,beta);
end
if l~=0
   FoundLValBelowLim=1;
end
