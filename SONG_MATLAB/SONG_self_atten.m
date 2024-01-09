function [tr,va,te] = SONG_self_atten(train,valid,test,patch_len)
    train_sample_num = size(train,2);
    test_sample_num = size(test,2);
    valid_sample_num = size(valid,2);
    patch_num = size(train,1)/patch_len;
    tr = [];
    te = [];
    va=[];
    shape = ones(patch_len,1);
    for train_sample_idx = 1:train_sample_num
        tr_sample = [];
        X = reshape(train(:,train_sample_idx),patch_len,patch_num);
        K = X;
        Q = X';
        M = Q*K/sqrt(size(K,1));
        A = softmax(M');
        for patch_idx = 1:patch_num
            Att = kron(A(:,patch_idx)',shape);
            Z = X.*Att; 
            S = sum(Z,2);
            tr_sample = [tr_sample;S];
        end
        tr = [tr tr_sample];
    end
    for valid_sample_idx = 1:valid_sample_num
        va_sample = [];
        X = reshape(valid(:,valid_sample_idx),patch_len,patch_num);
        K = X;
        Q = X';
        M = Q*K/sqrt(size(K,1));
        A = softmax(M');
        for patch_idx = 1:patch_num
            Att = kron(A(:,patch_idx)',shape);
            Z = X.*Att; 
            S = sum(Z,2);
            va_sample = [va_sample;S];
        end
        va = [va va_sample];
    end
    for test_sample_idx = 1:test_sample_num
        te_sample = [];
        X = reshape(test(:,test_sample_idx),patch_len,patch_num);
        K = X;
        Q = X';
        M = Q*K/sqrt(size(K,1));
        A = softmax(M');
        for patch_idx = 1:patch_num
            Att = kron(A(:,patch_idx)',shape);
            Z = X.*Att; 
            S = sum(Z,2);
            te_sample = [te_sample;S];
        end
        te = [te te_sample];
    end
end

