function [HiddernNeuronsNum,trainF,validF, testF]= SONG_PRF(X, Y, TX,ValidData,VaT, winSize, stride, dim,w,h,noiseLeve,actFun,para,lambda)
% X: N*d data
% Y: m*N label
% d is the sample dimension
% m is the class number
% N is the sample number
% dim is used to distinguish between table data and image data, and its value is 1 or 2
% w is the image width
% h is the image height
GRID = 8;
numsamples = size(X,1);
numdims = size(X,2);
HiddernNeuronsNum = winSize*4;
NumberOfTestSamples = size(TX,1);
NumberOfValidSamples = size(ValidData,1);
mask = 2*noiseLeve*rand(winSize,winSize)-noiseLeve;
trainF = [];
testF = [];
validF=[];
[~, label_index]=max(Y);%label_index is the position of the maximum value of each sample label

if dim == 2 % For image data
    X = reshape(X',h,w,numsamples);%Reconfigure X' into a three-dimensional array, h*w*n
    TX = reshape(TX',h,w,NumberOfTestSamples);
    ValidData = reshape(ValidData',h,w,NumberOfValidSamples);
%%     
    % Label Enhancement, used when PILer makes predictions
    % newY is the new label after label enhancement
    % ======================================================
    newY = zeros(GRID,size(Y,1)*GRID,size(Y,2));
    for j = 1:GRID
        for i = 1:size(Y,2)
            new_label_index = (label_index-1)*GRID+1;
            newY(j,new_label_index(i)+j-1,i) = 1;
        end
    end
    % ======================================================
    start_idx=1;
    stop = 0;

    while(~stop)
        if start_idx+winSize-1 >= w
                %break;
                end_idx  = w;
                stop = 1;
        else
            end_idx = start_idx+winSize-1;
        end
        len = end_idx - start_idx+1;
        hstart_idx=1;
        hstop = 0;
        while(~hstop)
            if hstart_idx+winSize-1 >= h
                %break;
                hend_idx  = h;
                hstop = 1;
            else
                hend_idx = hstart_idx+winSize-1;
            end
            hlen = hend_idx - hstart_idx+1;
            
            %%%%%%%%%%%%%%%
            map = X(hstart_idx:hend_idx,start_idx:end_idx,:);
            if stop == 1&&hstop==1 
                % map = [map;zeros(winSize-size(map,1),winSize-size(map,2),size(X,3))];%0 is added to the excess of the image
                map = [map;zeros(winSize-size(map,1),len,size(X,3))];%0 is added to the bottom
                %map = map;%without adding  reshape(map,hlen*len,numsamples)
                %break;%discard
            end
            %         if stop == 1&&hstop==0
            % %             map = [map;zeros(winSize,winSize-size(map,2),size(X,3))];
            %             break;
            %         end
            if stop == 0&&hstop == 1
                map = [map;zeros(winSize-size(map,1),winSize,size(X,3))];%0 is added to the excess of the image
            end
        
        
        
            if start_idx<=ceil(w/2)&&hstart_idx<=ceil(h/4)% upper left
                extY = reshape(newY(1,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx>ceil(w/2)&&hstart_idx<=ceil(h/4)%upper right
                extY = reshape(newY(2,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx<=ceil(w/2)&&hstart_idx>ceil(h/4)&&hstart_idx<=ceil(h/2)
                extY = reshape(newY(3,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx>ceil(w/2)&&hstart_idx>ceil(h/4)&&hstart_idx<=ceil(h/2)
                extY = reshape(newY(4,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx<=ceil(w/2)&&hstart_idx>ceil(h/2)&&hstart_idx<=ceil(h*3/4)
                extY = reshape(newY(5,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx>ceil(w/2)&&hstart_idx>ceil(h/2)&&hstart_idx<=ceil(h*3/4)
                extY = reshape(newY(6,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx<=ceil(w/2)&&hstart_idx>ceil(h*3/4)%lower left
                extY = reshape(newY(7,:,:),[size(Y,1)*GRID,size(Y,2)]);
            elseif start_idx>ceil(w/2)&&hstart_idx>ceil(h*3/4)%lower right
                extY = reshape(newY(8,:,:),[size(Y,1)*GRID,size(Y,2)]);
            end
        
            %map = reshape(map,winSize*winSize,numsamples); %Zeroes are added at the bottom right of the image
            map = reshape(map,len*winSize,numsamples);%Add zeros at the bottom of the image
            %map = reshape(map,hlen*len,numsamples);%not adding any zeros
%         extY = reshape(newY,[size(Y,1),size(Y,2)]);
            %newmap = zscore(map')';
            newmap = mapminmax(map',0,1)';
        %%
            method = 1;
            switch (method)
                case 1
                %#################### reduction #####################
                    l=1; 
                    while l<=1
                        InputDataLayer = newmap;

                        [net,F] = PILAE(InputDataLayer,HiddernNeuronsNum,actFun,2);%use 0.9* 行数 作为 hidden neural node number
                        %[net,F] = CP(InputDataLayer,trainBatchLabel,floor(p*size(InputDataLayer,1)),actFun,para);
                        ae{l} = net;
                        InputDataLayer = F; %when use PILAE
                        l = l+1;
                    end
                    trainF = [trainF;InputDataLayer];
                case 2
                %#################### prediction #####################
                    InputWeight=rand(HiddernNeuronsNum,len*winSize)*2-1;
                    BiasofHiddenNeurons=rand(HiddernNeuronsNum,1);
                    H=InputWeight*newmap;
                    ind=ones(1,numsamples);
                    BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    H=H+BiasMatrix;
                    %[HO,st] = mapminmax(H,0,1);
                    HO = ActivationFunc(H,actFun,para);
                    OutputWeight = extY*HO'/(HO*HO'+lambda*eye(HiddernNeuronsNum));
                    O=OutputWeight*HO;
                    trainF = [trainF;O];
                    clear HO O
            end
            
            %==================== Validation =================================
            map = ValidData(hstart_idx:hend_idx,start_idx:end_idx,:);
            if stop == 1&&hstop == 1
                %map = [map;zeros(winSize-size(map,1),winSize-size(map,2),size(X,3))];%0 is added to the excess of the image
                map = [map;zeros(winSize-size(map,1),len,size(ValidData,3))];%0 is added to the bottom
                %map = map;%without adding
                %break;%discard
            end
            %if stop == 1&&hstop==0
                %map = [map;zeros(winSize,winSize-size(map,2),size(TX,3))];
                % %break;
            %end
            if stop == 0&&hstop == 1
                map = [map;zeros(winSize-size(map,1),winSize,size(ValidData,3))];%0 is added to the bottom
            end
            map = reshape(map,len*winSize,NumberOfValidSamples);
            % map = reshape(map,winSize*winSize,NumberOfTestSamples);
            %map = reshape(map,hlen*len,NumberOfTestSamples);
            %newmap = zscore(map')';
            newmap = mapminmax(map',0,1)';
            switch (method)
                case 1
                %#################### reduction #####################
                    InputDataLayer = newmap;
                    l=1;
                    while l<=1
                        vF = ae{l}.WO'* InputDataLayer;% when use PILAE
                        %tF = ae{l}.WI* InputDataLayer;%when use CP;

                        vF = ActivationFunc(vF,actFun,2);

                        InputDataLayer = vF; %when use PILAE
                        %InputDataLayer = ae{l}.WO*tF;%when use CP;
                        l = l+1;
                    end
                    validF = [validF;InputDataLayer];
                case 2
                %#################### prediction #####################
                    H=InputWeight*newmap;
                    ind=ones(1,NumberOfValidSamples);
                    
                    BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    H=H+BiasMatrix;
                    %HO  =  mapminmax('apply',H,st);
                    HO = ActivationFunc(H,actFun,para);
                    VO=OutputWeight*HO;
                    %VY = ActivationFunc(VY,actFun,para);
                    validF = [validF;VO];
            end
            
            %==================== testing =================================
            map = TX(hstart_idx:hend_idx,start_idx:end_idx,:);
            if stop == 1&&hstop == 1
                %map = [map;zeros(winSize-size(map,1),winSize-size(map,2),size(X,3))];%0 is added to the excess of the image
                map = [map;zeros(winSize-size(map,1),len,size(TX,3))];%0 is added to the bottom
                %map = map;%without adding
                %break;%discard
            end
            %if stop == 1&&hstop==0
                %map = [map;zeros(winSize,winSize-size(map,2),size(TX,3))];
                % %break;
            %end
            if stop == 0&&hstop == 1
                map = [map;zeros(winSize-size(map,1),winSize,size(TX,3))];%0 is added to the bottom
            end
            map = reshape(map,len*winSize,NumberOfTestSamples);
            % map = reshape(map,winSize*winSize,NumberOfTestSamples);
            %map = reshape(map,hlen*len,NumberOfTestSamples);
            %newmap = zscore(map')';
            newmap = mapminmax(map',0,1)';
            switch (method)
                case 1
                %#################### reduction #####################
                    InputDataLayer = newmap;
                    l=1;
                    while l<=1
                        tF = ae{l}.WO'* InputDataLayer;% when use PILAE
                        %tF = ae{l}.WI* InputDataLayer;%when use CP;

                        tF = ActivationFunc(tF,actFun,2);

                        InputDataLayer = tF; %when use PILAE
                        %InputDataLayer = ae{l}.WO*tF;%when use CP;
                        l = l+1;
                    end
                    testF = [testF;InputDataLayer];
                case 2
                %#################### prediction #####################
                    H=InputWeight*newmap;
                    ind=ones(1,NumberOfTestSamples);
                    
                    BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    H=H+BiasMatrix;
                    %HO  =  mapminmax('apply',H,st);
                    HO = ActivationFunc(H,actFun,para);
                    TO=OutputWeight*HO;
                    %TY = ActivationFunc(TY,actFun,para);
                    testF = [testF;TO];
            end
            hstart_idx = hend_idx+1;
        end
        start_idx = end_idx+1;
    end
    clear newmap map;
else %for 1D data
    X = X';%dim*n
    %TX = reshape(TX',1,w,NumberOfTestSamples);
    TX = TX';
    ValidData = ValidData';
    HiddernNeuronsNum = 128;
    method = 1;%get prediction when it is 2, and reduction when it is 1
    GRID=2;
    newY = zeros(GRID,size(Y,1)*GRID,size(Y,2));
    for j = 1:GRID
        for i = 1:size(Y,2)
            new_label_index = (label_index-1)*GRID+1;
            newY(j,new_label_index(i)+j-1,i) = 1;
        end
    end
    
    start_idx=1;
    stop = 0;
    while(~stop)
        if start_idx+winSize-1 >= w
            %break;
            end_idx  = w;
            stop = 1;
        else
            end_idx = start_idx+winSize-1;
        end
        len = end_idx - start_idx+1;
        map = X(start_idx:end_idx,:);
        if stop == 1
            map = [map;zeros(start_idx+winSize-1-w,size(X,2))];%adding 0
            %map = map;
        end
        
        if start_idx<=ceil(w/2)
            extY = reshape(newY(1,:,:),[size(Y,1)*GRID,size(Y,2)]);
        elseif start_idx>ceil(w/2)
            extY = reshape(newY(2,:,:),[size(Y,1)*GRID,size(Y,2)]);
        end
        %extY = reshape(newY,[size(Y,1),size(Y,2)]);
        newmap = zscore(map')';
        %X = mapminmax(map,0,1)';
        %newmap = map;
        switch (method)
            case 1
                %#################### reduction #####################
                l=1; 
                while l<=1
                    InputDataLayer = newmap;

                    [net,F] = PILAE(InputDataLayer,HiddernNeuronsNum,actFun,2);
                    %[net,F] = CP(InputDataLayer,trainBatchLabel,floor(p*size(InputDataLayer,1)),actFun,para);
                    ae{l} = net;
                    InputDataLayer = F; %when use PILAE
                    l = l+1;
                end
                trainF = [trainF;InputDataLayer];
            case 2
                %#################### prediction #####################
                InputWeight=rand(HiddernNeuronsNum,winSize*winSize)*2-1;
                    %                     if HiddernNeuronsNum >= winSize*winSize
                    %                         InputWeight = orth(InputWeight);
                    %                     else
                    %                         InputWeight = orth(InputWeight')';
                    %                     end
                BiasofHiddenNeurons=rand(HiddernNeuronsNum,1);
                H=InputWeight*newmap;
                ind=ones(1,numsamples);
                BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                H=H+BiasMatrix;
                %[HO,st] = mapminmax(H,0,1);
                HO = ActivationFunc(H,actFun,para);
                OutputWeight = extY*HO'/(HO*HO'+lambda*eye(HiddernNeuronsNum));
                O=OutputWeight*HO;
                trainF = [trainF;O];         
        end
        
        %==================== Validation =================================
        map = ValidData(start_idx:end_idx,:);
        if stop == 1
            map = [map;zeros(start_idx+winSize-1-w,size(ValidData,2))];
            %map = map;%without adding
            %break;%discard
        end
        newmap = zscore(map')';
        newmap = mapminmax(map',0,1)';
        switch (method)
            case 1
                %#################### reduction #####################
                InputDataLayer = newmap;
                l=1;
                while l<=1
                    vF = ae{l}.WO'* InputDataLayer;% when use PILAE
                    %vF = ae{l}.WI* InputDataLayer;%when use CP;

                    vF = ActivationFunc(vF,actFun,2);

                    InputDataLayer = vF; %when use PILAE
                    %InputDataLayer = ae{l}.WO*vF;%when use CP;
                    l = l+1;
                end
                validF = [validF;InputDataLayer];
            case 2
                %#################### prediction #####################
                H=InputWeight*newmap;
                ind=ones(1,NumberOfValidSamples);
                    
                BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                H=H+BiasMatrix;
                %HO  =  mapminmax('apply',H,st);
                HO = ActivationFunc(H,actFun,para);
                VO=OutputWeight*HO;
                %VY = ActivationFunc(VY,actFun,para);
                validF = [validF;VO];
            end
        %==================== testing =================================
        map = TX(start_idx:end_idx,:);
        if stop == 1
            map = [map;zeros(start_idx+winSize-1-w,size(TX,2))];
        end
        newmap = zscore(map')';
        %newmap = map;
 
        InputDataLayer = newmap;
        switch (method)
            case 1
                %#################### reduction #####################
                l=1;
                while l<=1
                    tF = ae{l}.WO'* InputDataLayer;% when use PILAE
                    %tF = ae{l}.WI* InputDataLayer;%when use CP;

                    tF = ActivationFunc(tF,actFun,2);

                    InputDataLayer = tF; %when use PILAE
                    %InputDataLayer = ae{l}.WO*tF;%when use CP;
                    l = l+1;
                end
                testF = [testF;InputDataLayer];
            case 2
                %#################### prediction #####################
                H=InputWeight*newmap;
                ind=ones(1,NumberOfTestSamples);
                BiasMatrix=BiasofHiddenNeurons(:,ind);   %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                H=H+BiasMatrix;
                %HO  =  mapminmax('apply',H,st);
                HO = ActivationFunc(H,actFun,para);
                TO=OutputWeight*HO;
                %TY = ActivationFunc(TY,actFun,para);
                testF = [testF;TO];
        end     
        start_idx = end_idx+1;
    end
    clear newmap map;
end
    if method==2
        HiddernNeuronsNum = size(Y,1)*GRID;
    end
end

