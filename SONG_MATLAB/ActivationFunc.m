function H = ActivationFunc( tempH, ActivationFunction,p)
%ACTIVATIONFUNC Summary of this function goes here
%   Detailed explanation goes here
switch lower(ActivationFunction)%将字符串转换为小写
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-p.*tempH));
    case {'sin','sine'}
        %%%%%%%% Sine    傅里叶变换
        H = sin(tempH);
    case {'hardlim'}     %阈值型传递函数
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function   三角形径向基传输函数
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function   三角形径向基传输函数
        H = radbas(tempH);
    case {'gau'}
        %高斯分布，μ=0，sigma=1 标准正态分布 f(x)=1/(2*π)^2*e^(-x*x/2)
        H = 1./sqrt(2*pi)*exp(-1./2*(tempH.^2));
        %%%%%%%% ReLU
    case {'relu'}
        idx = find(tempH(:)<0);
        tempH(idx)=0;
        H = tempH;
    case {'srelu'}
        idx = find(tempH(:)<p);
        tempH(idx)=0;
        H = tempH;
    case {'tan'}
        H = tanh(p.*tempH);
    case {'prelu'}
        alpha = 0.02;
        idx = find(tempH(:)<0);
        tempH(idx)=alpha.*tempH(idx);
        H = tempH;
    case {'gelu'}
        H = tempH .* 1 ./ (1 + exp(-p.*tempH.*1.702));
    case {'mor'}
        H = cos(0.4.*tempH).*exp(-1./2*(tempH.^2));
end
end

