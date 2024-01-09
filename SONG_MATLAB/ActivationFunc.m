function H = ActivationFunc( tempH, ActivationFunction,p)
%ACTIVATIONFUNC Summary of this function goes here
%   Detailed explanation goes here
switch lower(ActivationFunction)%���ַ���ת��ΪСд
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-p.*tempH));
    case {'sin','sine'}
        %%%%%%%% Sine    ����Ҷ�任
        H = sin(tempH);
    case {'hardlim'}     %��ֵ�ʹ��ݺ���
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function   �����ξ�������亯��
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function   �����ξ�������亯��
        H = radbas(tempH);
    case {'gau'}
        %��˹�ֲ�����=0��sigma=1 ��׼��̬�ֲ� f(x)=1/(2*��)^2*e^(-x*x/2)
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

