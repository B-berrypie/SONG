clear;
clc;
%% Dataset
% Label is numbers
% % X is d*N matrix, N is the sample number; d is the dimension
% avg 91.49
Label = importdata('./Dataset/norbnumY.mat');
X = importdata('./Dataset/norbX.mat');
%% STRUCTURE
MAX_LAYER = 1;
MAX_SUBNET = 14;
TRAING_RATIO = 0.7;%for SPLIT_TRAIN_TEST=0
SAMPLE_RATIO = 0.6;% when there is only one sub-net, set it to 1.0
lambda=1e-7;
para = 0.05;
actFun = 'gelu';
p = 0.9;
nl = 0.0001;% conv noise level
SPLIT_TRAIN_TEST = 1; % 0:random split; 1: set the index
HIDDEN_NUM_Mode = 0;% 0: HIDDEN_NEURON_NUM; 1: n= p * dim;
HIDDEN_NEURON_NUM=3500;
randomProES = 0.99; % probability to early stop

randomProSN = 0.6; % probability of stopping add subnetwork

px = [];
py = [];


%%  Random seperate training and test set %%%%
if SPLIT_TRAIN_TEST == 0
    rand('seed',0);
    rand_idx= randperm(size(X,2));
    trainidx = rand_idx(1:ceil(size(X,2).*TRAING_RATIO));
    vaidx = rand_idx(ceil(size(X,2).*TRAING_RATIO)+1:ceil(size(X,2).*TRAING_RATIO)+1+ceil(size(X,2).*(1-TRAING_RATIO)/2));
    teidx = setdiff(1:size(X,2),[trainidx vaidx]);

elseif SPLIT_TRAIN_TEST == 1
    trainidx = 1:24300;
    vaidx = 20001:24299;
    teidx = 24301:48600;
end

%%  Normalization %%%%%%
Label = double(Label);
X = double(X);
% X = zscore(X')';
X = mapminmax(X,0,1);
%X = whiten(X')';

train_X = X(:,trainidx);
train_Y = Label(:,trainidx);

test_X = X(:,teidx);
test_Y = Label(:,teidx);

valid_X = X(:,vaidx);
valid_Y = Label(:,vaidx);

%%%% For confusion Matrix %%%%
[test_Y,i]= sort(test_Y);
test_X = test_X(:,i);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = train_X;
T = train_Y;

TT.P = test_X;
TT.T = test_Y;

VA.P = valid_X;
VA.T = valid_Y;


clear train_X test_X X Label;
NumberofTrainingData=size(P,2);
NumberofTestingData=size(TT.P,2);
NumberofValidationData=size(VA.P,2);
targetPrepro;
esbTY = zeros(size(TT.T));
esbVY = zeros(size(VA.T));
df_trainX = [];
df_trainY = [];
CM = zeros(size(TT.T,1),size(TT.T,1),MAX_SUBNET);
ind=2*ones(1,size(TT.T,1));
mask = diag(ind);
mask = mask-1;
LL = zeros(MAX_SUBNET,NumberofTestingData);
%% SONG
%  Laptop,Intel i5-8300H CPU @ 2.30GHz,16.0 GB RAM,Matlab R2018b
TrainingData = P';
TestData = TT.P';
ValidData = VA.P';
sub_net = 1;
best_V_Eacc = 0;
alltime = 0;
while true
    if (sub_net > MAX_SUBNET)
        break;
    end
    SR = normrnd(SAMPLE_RATIO,0.02);%increase sub-network diversity
    if SR > 1 || SR <=0
        SR = SAMPLE_RATIO;
    end
    %sampling
    [trainInd,valInd,testInd]=dividerand(NumberofTrainingData,SR,1-SR,0.0);
    trainBatch = TrainingData(trainInd,:);
    trainBatchLabel = T(:,trainInd);

    %PRF
    patch = randi([4 8])/3.5;%4-10
    PRFWinSize=ceil(32/patch);
    stride = ceil(PRFWinSize/1);
    tic;
    [hsize, trainBatch,validBatch,testBatch] = SONG_PRF(trainBatch,trainBatchLabel,TestData,ValidData,VA.T,PRFWinSize,stride,2,32,64,nl,actFun,para,lambda);
    if size(trainBatch,1)==0
        fprintf('No suitable Batch');
        continue;
    end

    [trainBatch,validBatch,testBatch] = SONG_self_atten(trainBatch,validBatch,testBatch,hsize);

    trainBatch = zscore(trainBatch')';
    validBatch = zscore(validBatch')';
    testBatch = zscore(testBatch')';
    %======================================
    %training set
    l=1; %layer for PILer-MLP
    InputDataLayer = trainBatch;
    best_V_acc = 0;
    TInputDataLayer=[];
    while l<=MAX_LAYER
        [net,F] = PILAE(InputDataLayer,floor(p*size(InputDataLayer,1)),actFun,2);%use 0.9* size(trainBatch,1) as hidden neural node number
        %[net,F] = CP(InputDataLayer,trainBatchLabel,floor(p*size(InputDataLayer,1)),actFun,para);
        ae{l} = net;
        InputDataLayer = F;
        TInputDataLayer{l}=InputDataLayer;
        %  ======= PILer-Classifier  Training ==========
        numsamples = size(InputDataLayer,2);
        numdims = size(InputDataLayer,1);
        po = 3+(6-3)*rand;
        if HIDDEN_NUM_Mode == 1
            HiddernNeuronsNum = floor(po*numdims);
        else
            HiddernNeuronsNum = HIDDEN_NEURON_NUM;
        end
        InputWeight=rand(HiddernNeuronsNum,numdims)*2-1;
        if HiddernNeuronsNum >= numdims
            InputWeight = orth(InputWeight);
        else
            InputWeight = orth(InputWeight')';
        end
        BiasofHiddenNeurons=rand(HiddernNeuronsNum,1);
        cls{l}.Bias=BiasofHiddenNeurons;
        tempH=InputWeight*InputDataLayer;
        cls{l}.IW = InputWeight;
        ind=ones(1,numsamples);
        BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
        tempH=tempH+BiasMatrix;
        %[HO,st] = mapminmax(tempH,0,1);
        HO = ActivationFunc(tempH,actFun,para);
        OutputWeight = trainBatchLabel*HO'/(HO*HO'+lambda*eye(HiddernNeuronsNum));
        Y=OutputWeight*HO;
        cls{l}.OW = OutputWeight;
        clear HO tempH;
        [~, label_index_expected]=max(trainBatchLabel);
        [~,label_index_actual]=max(Y);
        MissClassification=length(find(label_index_actual~=label_index_expected));
        acc=1-MissClassification/size(trainBatchLabel,2);
        %fprintf('Classification %d: Training accuracy %.4f\n',sub_net,acc);
        l = l+1;
    end
    %validating set======Determine the number of MLP layers================
    InputDataLayer = validBatch;
    l=1;
    while l<=MAX_LAYER
        vF = ae{l}.WO'* InputDataLayer;% when use PILAE
        %vF = ae{l}.WI* InputDataLayer;%when use CP;
        
        vF = ActivationFunc(vF,actFun,2);
        
        InputDataLayer = vF; %when use PILAE
        %InputDataLayer = ae{l}.WO*vF;%when use CP;
        VY=0;
        [pred,Vacc] = SONG_Test_Classifier(InputDataLayer,VA.T,VY,cls{l}.IW,cls{l}.OW,cls{l}.Bias,para,actFun);
        if Vacc>best_V_acc
            best_V_acc = Vacc;
            validBatch = InputDataLayer;
            MAX_LAYER2 = MAX_LAYER;
        else
            rand('seed',sum(clock));
            if rand(1) < randomProES
                MAX_LAYER2 = l-1;
                %disp(['Test_LAYER£º' num2str(MAX_LAYER2)]);
                break;
            end
        end
        l = l+1;
    end
    alltime=alltime+toc;
    %testing set
    InputDataLayer = testBatch;
    l=1;
    while l<=MAX_LAYER2
        tF = ae{l}.WO'* InputDataLayer;% when use PILAE
        %tF = ae{l}.WI* InputDataLayer;%when use CP;
        
        tF = ActivationFunc(tF,actFun,2);
        
        InputDataLayer = tF; %when use PILAE
        %InputDataLayer = ae{l}.WO*tF;%when use CP;
        l = l+1;
    end
    clear tF vF;
    testBatch = InputDataLayer;
    %======================================
    
    %  ======= PILer-Classifier  validating  For the determination of the number of sub-networks==========
    [pred,VEacc,esbVY] = SONG_Test_Classifier(validBatch,VA.T,esbVY,cls{MAX_LAYER2}.IW,cls{MAX_LAYER2}.OW,cls{MAX_LAYER2}.Bias,para,actFun);
    %if mod(sub_net,2)==0
        %fprintf('====== %d Ensemble valid accuracy %.4f ======\n',sub_net,VEacc);
    %end
    if VEacc >= best_V_Eacc
        best_V_Eacc = VEacc;
        %fprintf('best_V_Eacc %.4f\n',best_V_Eacc);
    else
        rand('seed',sum(clock));
        if rand(1) < randomProSN
            sub_net = sub_net + 1;
            break;
            %continue;
        end
    end
    
    %  ======= PILer-Classifier  Testing ==========
    [pred,Eacc,esbTY] = SONG_Test_Classifier(testBatch,TT.T,esbTY,cls{MAX_LAYER2}.IW,cls{MAX_LAYER2}.OW,cls{MAX_LAYER2}.Bias,para,actFun);
    if mod(sub_net,3)==0
        fprintf('====== %d Ensemble Test accuracy %.4f ======\n',sub_net,Eacc);
    end
    h = animatedline('Color','b','LineWidth',2);
    px = [px sub_net];
    py = [py Eacc];
    for k = 1:length(px)
        addpoints(h,px(k),py(k));
    end
    ylabel('Accuracy');
    xlabel('Subnet');
    %ylim([0 1]);
    grid on
    pause(0.001)
    sub_net = sub_net + 1;
end

[~, label_index_expected]=max(TT.T);
[~,label_index_actual]=max(esbTY);
MissClassification=length(find(label_index_actual~=label_index_expected));
Eacc=1-MissClassification/size(TT.T,2);
fprintf('====== Final %d ensemble Test accuracy %.4f ======\n',sub_net-1,Eacc);
disp( ['running time: ',num2str(alltime) ] );
