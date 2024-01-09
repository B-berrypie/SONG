%%%%%%%%%%%% Preprocessing the data of classification
sorted_target=sort(cat(2,T,TT.T),2);
label=zeros(1,1); %   Find and save in 'label' class label from training and testing data sets
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:(NumberofTrainingData+NumberofTestingData)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;
        label(1,j) = sorted_target(1,i);
    end
end
number_class=j;
NumberofOutputNeurons=number_class;
%%%%%%%%%% Processing the targets of training
temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == T(1,i)
            break;
        end
    end
    temp_T(j,i)=1;
end
T=temp_T*2-1;
%T=temp_T;
%%%%%%%%%% Processing the targets of testing
temp_T_T=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == TT.T(1,i)
            break;
        end
    end
    temp_T_T(j,i)=1;
end
TT.T=temp_T_T*2-1;
%TT.T=temp_T_T;
%%%%%%%%%% Processing the targets of validation
temp_V_T=zeros(NumberofOutputNeurons, NumberofValidationData);
for i = 1:NumberofValidationData
    for j = 1:number_class
        if label(1,j) == VA.T(1,i)
            break;
        end
    end
    temp_V_T(j,i)=1;
end
VA.T=temp_V_T*2-1;
clear temp_V_T temp_T temp_V_T;