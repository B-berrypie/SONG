function [net F] = PILAE(X,p,actFun,para)
    %   X = d*N
    %   d is the sample dimension
    %   N is the sample number
    %   p is the hidden neural node number
    lambda = 1e-8;
    N = size(X,2);
    d = size(X,1);
    tic
    %%%%%%%%%%%%%%%%%
    %R = eye(p,N);%p*N
    %X_pinv=pinv(X);%N*d
    %WI = R*X_pinv;%p*d
    %%%%%%%%%%%%%%%%%
    WI = (rand(p,d)*2-1);
    if p >= d
        WI = orth(WI);
    else
        WI = orth(WI')';
    end
    %%%%%%%%%%%%%%%%%%%%%

    H = WI*X;%p*N
    HO = ActivationFunc(H,actFun,para);
    %HO = mapminmax(H')';
    %WO = Y*pinv(HO);d*p
    WO = X*HO'/(HO*HO'+lambda*eye(p));%d*p
    %O = WO*HO;%m*N
    %WO = max(WO-thrd,0) - max(-WO-thrd,0);
    F = WO'*X;
    F = ActivationFunc(F,actFun,para);
    trainingTime = toc;
    %fprintf('======Classification training time: %.4f =======\n',trainingTime);
    net.WI = WI;
    net.WO = WO;
    %net.st = st;
end

