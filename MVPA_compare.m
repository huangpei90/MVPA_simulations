% Compare MVPA metrics. Requirements:
% * SPM 12 - https://www.fil.ion.ucl.ac.uk/spm/software/spm12
% * covdiag.m - https://github.com/jooh/pilab
%
% If using in your own work, please cite
% Huang, P., Carlin, J. D., Alink, A., Kriegeskorte, N., Henson, R.N.A, & Correia, M.M.
% Prospective Motion Correction Improves the Sensitivity of FMRI Pattern Decoding.
% Human Brain Mapping, 39, 4018–31. https://doi.org/10.1002/hbm.24228.
function MVPA_compare()

params.beta_std=1;                          % Std in beta values across voxels

x = 2/sqrt(pi);                             % real contrast

params.CNR= 0.05:0.001:0.1;                 % CNR range
params.noisemat=x./params.CNR;              % Standard deviation of noise introduced
params.thermal_noise = 4;                   % magnitude of thermal noise
params.physio_vects = 10;                   % number of physiological vectors being simulated

%Declare GLM for this study
glm_var.TR = 2;                             % TR
glm_var.blocks = 4;                         % no of orientation blocks
glm_var.blockdur = 8;                       % block dur in terms of TR
glm_var.restdur = 4;                        % rest dur between blocks in terms of TR
glm_var.n_subruns = 4;                      % no of subruns


params.nvox = 500;
niter = 1000;


% custom seed but preserve deterministic behaviour over iterations
seed = 123456;

glm = create_glm(glm_var);

models = struct('design',{glm.design.design_mat,glm.design.SVM_mat},...
    'traincontrast',{[1 -1],repmat([repmat(1,1,glm_var.blocks) repmat(-1,1,glm_var.blocks)],1,(glm_var.n_subruns-1))},...
    'testcontrast',{[1,-1],[repmat(1,1,glm_var.blocks) repmat(-1,1,glm_var.blocks)]},...
    'combine', {@vertcat, @blkdiag}, ...
    'name',{'bycon','byblock'});

weights{1} = @calcSVM;
weights{2} = @calcLDC;
weights{3} = @calcEUC;

parfor iter = 1:niter
    rng(iter * seed);
    for sub = 1:15
        noiseless_data =  gen_subj(glm, params);
        for k=1:size(params.noisemat,2)
            % add_noise adds both thermal and physiological noise
            % add_tnoise adds only thermal noise with magnitude noisemat(k)
            % add_noise with params.thermal_noise = 0 adds only physiological noise
            data = add_noise(noiseless_data, params, k);
            estimates_block = est_calc(data, weights, models(2));        % 1 regressor per block
            estimates_cond = est_calc(data, weights, models(1));         % 1 regressor per condition
            results(iter).estimates_cond(sub,k) = estimates_cond;
            results(iter).estimates_block(sub,k) = estimates_block;
        end
    end
end

save('results.mat','results','params')


end

function glm = create_glm(glm_var)
    glm.subrun_length = 2*glm_var.restdur+2*glm_var.blocks*(glm_var.blockdur+glm_var.restdur); 
    glm.hrf_distri = spm_hrf(glm_var.TR);
    % Construct GLM matrix 
    subrun = cell(glm_var.n_subruns,1);
    design_mat = cell(glm_var.n_subruns,1);
    SVM_mat = cell(glm_var.n_subruns,1);
    for k = 1:glm_var.n_subruns   
        design = zeros(glm.subrun_length,2);
        SVM_design = zeros(glm.subrun_length,2*glm_var.blocks);
        if k==1 || k==4
            subrun{k} = repmat([1 2],1,glm_var.blocks);
        else
            subrun{k} = repmat([2 1],1,glm_var.blocks);
        end
        %Define start times for each subrun and input into design matrix
        sr1_start = glm_var.restdur+1;

        cur_timepoint = sr1_start;
        for i=1:size(subrun{k},2)
            design(cur_timepoint:cur_timepoint+glm_var.blockdur-1,subrun{k}(i))=1;
            cur_timepoint = cur_timepoint+glm_var.blockdur+glm_var.restdur;
        end
        %convolve with HRF to generate the design matrix
        design_mat{k}=conv2(design,glm.hrf_distri,'same');


        %Generate individual columns for SVM classification
        cur_timepoint = sr1_start;
        c1_count = 1;
        c2_count = 1;
        for i=1:length(subrun{k})
            if subrun{k}(i)==1
                SVM_design(cur_timepoint:cur_timepoint+glm_var.blockdur-1,c1_count)=1;
                c1_count = c1_count+1;
            else
                SVM_design(cur_timepoint:cur_timepoint+glm_var.blockdur-1,glm_var.blocks+c2_count)=1;
                c2_count = c2_count+1;

            end
            cur_timepoint = cur_timepoint+glm_var.blockdur+glm_var.restdur;
        end
        %convolve with HRF to generate the design matrix
        SVM_mat{k} = conv2(SVM_design,glm.hrf_distri,'same');
    end

    glm.design.design_mat = design_mat;
    glm.design.SVM_mat = SVM_mat;
end


function noiseless_data = gen_subj(glm,params)
betas = normrnd(0,params.beta_std,2,params.nvox);
noiseless_data = cell(size(glm.design.SVM_mat,1),1);
for i=1:size(glm.design.design_mat,1)
    noiseless_data{i} = glm.design.design_mat{i}*betas;
end

end

function data = add_noise(noiseless_data, params,k)
    data = cell(size(noiseless_data));
    noise = params.noisemat(k);
    thermal_noise = params.thermal_noise;
    physio_noise = noise-thermal_noise;
    noise_vect=normrnd(0,1,params.physio_vects,size(noiseless_data{1},2));       % physio noise consistent across voxels
    for i=1:size(noiseless_data,1)
        noise_proj=normrnd(0,1,size(noiseless_data{i},1),params.physio_vects);   % projection vector of physio noise vector on data
        physio_rnd = noise_proj*noise_vect;
        physio_rnd = physio_rnd/sqrt(params.physio_vects);
        thermal_rnd = normrnd(0,1,size(noiseless_data{i},1),size(noiseless_data{i},2));
        total_noise = physio_noise*physio_rnd+thermal_noise*thermal_rnd;
        data{i} = noiseless_data{i} + total_noise;
    end
end



function data = add_tnoise(noiseless_data, params,k)
    data = cell(size(noiseless_data));
    noise = params.noisemat(k);
    for i=1:size(noiseless_data,1)
        total_noise= normrnd(0,noise,size(noiseless_data{i},1),size(noiseless_data{i},2));
        data{i} = noiseless_data{i} + total_noise;
    end
end

function estimates = est_calc(data,weights,models)

    weight_names = {'SVM_dist','LDC_dist','EUC_dist','SVM_class'};

    beta_model = models.design;
    train_contrast = models.traincontrast;
    test_contrast = models.testcontrast;
    
    final_data = detrend(data);
    final_design = detrend(beta_model);
    
    %loop over all 4 iterations
    for i=1:4
        trainind = setdiff(1:4,i);

        train_design = models.combine(final_design{trainind});
        train_data = vertcat(final_data{trainind});

        test_design = final_design{i};
        test_data = final_data{i};
 
        
        for j=1:3
            weight_vect = weights{j}(train_design, train_data, train_contrast);
            test_est = test_design \ test_data;
            test_con = test_contrast * test_est;
            estimates.(weight_names{j})(i) = test_con * weight_vect';
        end
        
        % SVM Classifier
        train_contrast(train_contrast==-1) = 0;
        test_contrast(test_contrast==-1) = 0;
        train_betas = train_design \ train_data;
        test_betas = test_design \ test_data;
        svm_model = fitcsvm(train_betas,train_contrast);
        y_predict = predict(svm_model,test_betas);
        estimates.SVM_class(i) = mean(1-abs(y_predict-test_contrast'));
    end
    for i=1:4
        estimates.(weight_names{i}) = mean(estimates.(weight_names{i}));
    end
end

function detrended_data = detrend(data)
    detrended_data = cell(size(data,1),1);
    for i=1:size(data,1)
        % Linear, first order sinusoidal and mean detrending
        nvols = size(data{i},1);
        linear_trend=1:nvols;
        sin_trend=sin((linear_trend)*2*pi/(nvols-1));
        cos_trend=cos((linear_trend)*2*pi/(nvols-1));
        mean_trend = ones(1,nvols);
        dt_design = [linear_trend;sin_trend;cos_trend;mean_trend]';

        trend = dt_design\data{i};
        est = dt_design*trend;
        detrended_data{i} = data{i} - est;
    end
end

function weights_vector = calcSVM(train_design,train_data,train_contrast)
    train_contrast(train_contrast==-1) = 0;
    train_est = train_design \ train_data;
    model = fitcsvm(train_est,train_contrast);
    weights_vector=model.Alpha'*model.SupportVectors;
end

function weights_vector = calcLDC(train_design,train_data,train_contrast) 
    train_est = train_design \ train_data; 
    train_con = train_contrast * train_est;
    
    train_predict = train_design*train_est;
    train_res = train_data - train_predict;
    
    cd_cov = covdiag(train_res);
    cd_weights2 = train_con / cd_cov;
    weights_vector = cd_weights2;
end

function weights_vector = calcEUC(train_design,train_data,train_contrast)
    train_est = train_design \ train_data; 
    train_con = train_contrast * train_est;
    weights_vector = train_con;
end
