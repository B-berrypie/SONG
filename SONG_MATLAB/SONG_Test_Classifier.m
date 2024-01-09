function [label_index_actual,Eacc,esbTY] = SONG_Test_Classifier(data,label,esbTY,InputWeight,OutputWeight,BiasofHiddenNeurons,para,actFun)
    InputDataLayer = data';
    InputLabel = label;
    px = [];
    py = [];
    NumberOfTestSamples = size(InputDataLayer,1);
    tempH=InputWeight*InputDataLayer';
    ind=ones(1,NumberOfTestSamples);
    clear InputDataLayer;
    BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    tempH=tempH+BiasMatrix;
    %HO = mapminmax(tempH,0,1);
    %HO  =  mapminmax('apply',tempH,st);
    HO = ActivationFunc(tempH,actFun,para);
    TY=OutputWeight*HO;
    
    %TY =  ActivationFunc(TY,'sig',10);
    TY = mapminmax(TY',0,1)';
    TY = TY./sum(TY,1);
    
    [~, label_index_expected]=max(InputLabel);
    [~,label_index_actual]=max(TY);
    MissClassification=length(find(label_index_actual~=label_index_expected));
    Tacc=1-MissClassification/size(InputLabel,2);
    %fprintf('Classification %d: Test accuracy %.4f\n',sub_net,Tacc);
    esbTY = esbTY + TY;
    
    [~, label_index_expected]=max(InputLabel);
    [~,label_index_actual]=max(esbTY);
    MissClassification=length(find(label_index_actual~=label_index_expected));
    Eacc=1-MissClassification/size(InputLabel,2);
end   