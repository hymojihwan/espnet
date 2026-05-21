function CHiME3_simulate_data_patched_parallel(official,nj,chime4_dir,chime3_dir,out_dir,sets_filter)
    % Optional 6th arg: 'all' (default) = tr05+dt05+et05; 'dt05_only' = valid only (faster re-run).

    if nargin < 6, sets_filter = 'all'; end

    % CHIME3_SIMULATE_DATA Creates simulated data for the 3rd CHiME Challenge
    %
    % CHiME3_simulate_data
    % CHiME3_simulate_data(official)
    %
    % Input:
    % official: boolean flag indicating whether to recreate the official
    % Challenge data (default) or to create new (non-official) data
    %
    % If you use this software in a publication, please cite:
    %
    % Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
    % third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
    % task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
    % and Understanding Workshop (ASRU), 2015.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
    %                Inria (Emmanuel Vincent)
    %                Mitsubishi Electric Research Labs (Shinji Watanabe)
    % This software is distributed under the terms of the GNU Public License
    % version 3 (http://www.gnu.org/licenses/gpl.txt)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    utils_folder = sprintf('%s/tools/utils', chime4_dir);
    enhancement_folder = sprintf('%s/tools/enhancement/', chime3_dir);
    addpath(utils_folder,'-end');
    addpath(enhancement_folder);
    sim_folder = sprintf('%s/tools/simulation', chime4_dir);
    addpath(sim_folder);
    upath = sprintf('%s/data/audio/16kHz/isolated/', chime4_dir);
    cpath = sprintf('%s/data/audio/16kHz/embedded/', chime4_dir);
    bpath = sprintf('%s/data/audio/16kHz/backgrounds/', chime4_dir);
    apath = sprintf('%s/data/annotations/', chime4_dir);
    upath_ext = 'local/nn-gev/data/audio/16kHz/isolated_ext/';
    upath_simu = 'local/nn-gev/data/audio/16kHz/isolated/';
    % 1ch track: process and write only one channel (faster). Set nchan=6 and out_channels=[1:6] for full 6ch.
    nchan = 1;
    out_channels = 1;   % index 1 in our 1ch data
    ref_ch = 1;         % physical channel to read from embedded/backgrounds when nchan==1 (CH1)

    % SNR control: train/valid = random SNR in [min,max] dB; test = baseline 0 dB + per-SNR (dB)
    snr_train_range = [-5 15];  % train and dt05 (valid): random SNR in this range when mixing clean + noise
    snr_eval_list = [-5 0 5 10 15];  % et05 (test): per-SNR evaluation sets

    % Define hyper-parameters
    pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
    wlen_sub=256; % STFT window length in samples
    blen_sub=4000; % average block length in samples for speech subtraction (250 ms)
    ntap_sub=12; % filter length in frames for speech subtraction (88 ms)
    wlen_add=1024; % STFT window length in samples for speaker localization
    del=-3; % minimum delay (0 for a causal filter)

    %% Create simulated training dataset from original WSJ0 data %%
    if exist('equal_filter.mat','file'),
        load('equal_filter.mat');
    else
        % Compute average power spectrum of booth data
        nfram=0;
        bth_spec=zeros(wlen_sub/2+1,1);
        sets={'tr05' 'dt05'};
        for set_ind=1:length(sets),
            set=sets{set_ind};
            mat=json2mat([apath set '_bth.json']);
            for utt_ind=1:length(mat),
                oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
            fprintf('%s\n',[upath set '_bth/' oname '.CH0.wav']);
                o=audioread([upath set '_bth/' oname '.CH0.wav']);
                O=stft_multi(o.',wlen_sub);
                nfram=nfram+size(O,2);
                bth_spec=bth_spec+sum(abs(O).^2,2);
            end
        end
        bth_spec=bth_spec/nfram;

        % Compute average power spectrum of original WSJ0 data
        nfram=0;
        org_spec=zeros(wlen_sub/2+1,1);
        olist=dir([upath 'tr05_org/*.wav']);
        for f=1:length(olist),
            oname=olist(f).name;
            o=audioread([upath 'tr05_org/' oname]);
            O=stft_multi(o.',wlen_sub);
            nfram=nfram+size(O,2);
            org_spec=org_spec+sum(abs(O).^2,2);
        end
        org_spec=org_spec/nfram;

        % Derive equalization filter
        equal_filter=sqrt(bth_spec./org_spec);
        save('equal_filter.mat','equal_filter');
    end
    % Read official annotations
    if official,
        mat=json2mat([apath 'tr05_simu.json']);
    % Create new (non-official) annotations
    else
        mat=json2mat([apath 'tr05_org.json']);
        ir_mat=json2mat([apath 'tr05_real.json']);
        for utt_ind=1:length(mat),
            oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_ORG'];
            osize=audioread([upath 'tr05_org/' oname '.wav'],'size');
            dur=osize(1)/16000;
            envirs={'BUS' 'CAF' 'PED' 'STR'};
            envir=envirs{randperm(4,1)}; % draw a random environment
            mat{utt_ind}.environment=envir;
            blist=dir([bpath '*' envir '.CH1.wav']);
            dur_diff=inf(1,length(ir_mat));
            for ir_ind=1:length(ir_mat),
                if strcmp(ir_mat{ir_ind}.environment,envir),
                    ir_dur=ir_mat{ir_ind}.end-ir_mat{ir_ind}.start;
                    dur_diff(ir_ind)=abs(ir_dur-dur);
                end
            end
            ir_ind=find(isinf(dur_diff));
            ir_ind=ir_ind(1);
            nfail=true;
            while nfail,
                bname=blist(randperm(length(blist),1)).name(1:end-8); % draw a random background recording
                mat{utt_ind}.noise_wavfile=bname;
                bsize=audioread([bpath bname '.CH1.wav'],'size');
                bdur=bsize(1)/16000;
                mat{utt_ind}.noise_start=ceil(rand(1)*(bdur-dur)*16000)/16000; % draw a random time
                mat{utt_ind}.noise_end=mat{utt_ind}.noise_start+dur;
                nname=mat{utt_ind}.noise_wavfile;
                nbeg=round(mat{utt_ind}.noise_start*16000)+1;
                nend=round(mat{utt_ind}.noise_end*16000);
                n=zeros(nend-nbeg+1,nchan);
                for c=1:nchan,
                    ch = ref_ch*(nchan==1) + c*(nchan>1);
                    n(:,c)=audioread([bpath nname '.CH' int2str(ch) '.wav'],[nbeg nend]);
                end
                npow=sum(n.^2,1);
                npow=10*log10(npow/max(npow));
                nfail=any(npow<=pow_thresh); % check for microphone failure
            end
            xfail=true;
            while xfail,
                dur_diff(ir_ind)=inf;
                [~,ir_ind]=min(dur_diff); % pick impulse response from the same environment with the closest duration
                if dur_diff(ir_ind)==inf,
                    keyboard;
                end
                mat{utt_ind}.ir_wavfile=ir_mat{ir_ind}.wavfile;
                mat{utt_ind}.ir_start=ir_mat{ir_ind}.start;
                mat{utt_ind}.ir_end=ir_mat{ir_ind}.end;
                iname=mat{utt_ind}.ir_wavfile;
                ibeg=round(mat{utt_ind}.ir_start*16000)+1;
                iend=round(mat{utt_ind}.ir_end*16000);
                x=zeros(iend-ibeg+1,nchan);
                for c=1:nchan,
                    ch = ref_ch*(nchan==1) + c*(nchan>1);
                    x(:,c)=audioread([cpath iname '.CH' int2str(ch) '.wav'],[ibeg iend]);
                end
                xpow=sum(x.^2,1);
                xpow=10*log10(xpow/max(xpow));
                xfail=any(xpow<=pow_thresh); % check for microphone failure
            end
            mat{utt_ind}=orderfields(mat{utt_ind});
        end
        mat2json(mat,[apath 'tr05_simu_new.json']);
    end

    use_parfor = true;
    try
        p = parpool('local', nj);
    catch ME
        use_parfor = false;
        p = [];
        warning('Parallel Computing Toolbox not available (%s). Running in serial.', ME.message);
    end

    % Loop over utterances (skip tr05 when only re-running valid)
    if ~strcmp(sets_filter, 'dt05_only')
    if use_parfor
        parfor utt_ind=1:length(mat),
        if official,
            udir=[upath_simu 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
            udir_ext=[upath_ext 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
        else
            udir=[upath 'tr05_' lower(mat{utt_ind}.environment) '_simu_new/'];
            udir_ext=[upath_ext 'tr05_' lower(mat{utt_ind}.environment) '_simu_new/'];
        end
        if ~exist(udir,'dir'),
            system(['mkdir -p ' udir]);
        end
        if ~exist(udir_ext,'dir'),
            system(['mkdir -p ' udir_ext]);
        end
        oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_ORG'];
        iname=mat{utt_ind}.ir_wavfile;
        nname=mat{utt_ind}.noise_wavfile;
        uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
        ibeg=round(mat{utt_ind}.ir_start*16000)+1;
        iend=round(mat{utt_ind}.ir_end*16000);
        nbeg=round(mat{utt_ind}.noise_start*16000)+1;
        nend=round(mat{utt_ind}.noise_end*16000);

        % Load WAV files
        o=audioread([upath 'tr05_org/' oname '.wav']);
        [r,fs]=audioread([cpath iname '.CH0.wav'],[ibeg iend]);
        x=zeros(iend-ibeg+1,nchan);
        n=zeros(nend-nbeg+1,nchan);
        for c=1:nchan,
            ch = ref_ch*(nchan==1) + c*(nchan>1);
            x(:,c)=audioread([cpath iname '.CH' int2str(ch) '.wav'],[ibeg iend]);
            n(:,c)=audioread([bpath nname '.CH' int2str(ch) '.wav'],[nbeg nend]);
        end

        % Compute the STFT (short window)
        O=stft_multi(o.',wlen_sub);
        R=stft_multi(r.',wlen_sub);
        X=stft_multi(x.',wlen_sub);

        % Estimate 88 ms impulse responses on 250 ms time blocks
        A=estimate_ir(R,X,blen_sub,ntap_sub,del);

        % Train: mix clean and noise at a random SNR (ratio) in [-5, 15] dB per utterance
        SNR_dB = snr_train_range(1) + (snr_train_range(2)-snr_train_range(1))*rand();
        SNR = 10^(SNR_dB/10);

        % Equalize microphone
        [~,nfram]=size(O);
        O=O.*repmat(equal_filter,[1 nfram]);
        o=istft_multi(O,nend-nbeg+1).';

        % Compute the STFT (long window)
        O=stft_multi(o.',wlen_add);
        X=stft_multi(x.',wlen_add);
        [nbin,nfram] = size(O);

        % Localize and track the speaker
        [~,TDOAx]=localize(X,[1:nchan]);

        % Interpolate the spatial position over the duration of clean speech
        TDOA=zeros(nchan,nfram);
        for c=1:nchan,
            TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
        end

        % Filter clean speech
        Ysimu=zeros(nbin,nfram,nchan);
        for f=1:nbin,
            for t=1:nfram,
                Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
                Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
            end
        end
        ysimu=istft_multi(Ysimu,nend-nbeg+1).';

        % Scale clean so mixture has power ratio SNR (speech/noise), then add: noisy = clean_scaled + noise
        ysimu=sqrt(SNR/sum(sum(ysimu.^2))*sum(sum(n.^2)))*ysimu;
        xsimu=ysimu+n;

        % Write WAV file (1ch track: out_channels only)
        for c=out_channels,
            audiowrite([udir uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
            audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:, c),fs);
            audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu(:, c), fs);
        end
    end
    else
        for utt_ind=1:length(mat),
        if official,
            udir=[upath_simu 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
            udir_ext=[upath_ext 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
        else
            udir=[upath 'tr05_' lower(mat{utt_ind}.environment) '_simu_new/'];
            udir_ext=[upath_ext 'tr05_' lower(mat{utt_ind}.environment) '_simu_new/'];
        end
        if ~exist(udir,'dir'),
            system(['mkdir -p ' udir]);
        end
        if ~exist(udir_ext,'dir'),
            system(['mkdir -p ' udir_ext]);
        end
        oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_ORG'];
        iname=mat{utt_ind}.ir_wavfile;
        nname=mat{utt_ind}.noise_wavfile;
        uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
        ibeg=round(mat{utt_ind}.ir_start*16000)+1;
        iend=round(mat{utt_ind}.ir_end*16000);
        nbeg=round(mat{utt_ind}.noise_start*16000)+1;
        nend=round(mat{utt_ind}.noise_end*16000);

        % Load WAV files
        o=audioread([upath 'tr05_org/' oname '.wav']);
        [r,fs]=audioread([cpath iname '.CH0.wav'],[ibeg iend]);
        x=zeros(iend-ibeg+1,nchan);
        n=zeros(nend-nbeg+1,nchan);
        for c=1:nchan,
            ch = ref_ch*(nchan==1) + c*(nchan>1);
            x(:,c)=audioread([cpath iname '.CH' int2str(ch) '.wav'],[ibeg iend]);
            n(:,c)=audioread([bpath nname '.CH' int2str(ch) '.wav'],[nbeg nend]);
        end

        % Compute the STFT (short window)
        O=stft_multi(o.',wlen_sub);
        R=stft_multi(r.',wlen_sub);
        X=stft_multi(x.',wlen_sub);

        % Estimate 88 ms impulse responses on 250 ms time blocks
        A=estimate_ir(R,X,blen_sub,ntap_sub,del);

        % Train: mix clean and noise at a random SNR (ratio) in [-5, 15] dB per utterance
        SNR_dB = snr_train_range(1) + (snr_train_range(2)-snr_train_range(1))*rand();
        SNR = 10^(SNR_dB/10);

        % Equalize microphone
        [~,nfram]=size(O);
        O=O.*repmat(equal_filter,[1 nfram]);
        o=istft_multi(O,nend-nbeg+1).';

        % Compute the STFT (long window)
        O=stft_multi(o.',wlen_add);
        X=stft_multi(x.',wlen_add);
        [nbin,nfram] = size(O);

        % Localize and track the speaker
        [~,TDOAx]=localize(X,[1:nchan]);

        % Interpolate the spatial position over the duration of clean speech
        TDOA=zeros(nchan,nfram);
        for c=1:nchan,
            TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
        end

        % Filter clean speech
        Ysimu=zeros(nbin,nfram,nchan);
        for f=1:nbin,
            for t=1:nfram,
                Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
                Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
            end
        end
        ysimu=istft_multi(Ysimu,nend-nbeg+1).';

        % Scale clean so mixture has power ratio SNR (speech/noise), then add: noisy = clean_scaled + noise
        ysimu=sqrt(SNR/sum(sum(ysimu.^2))*sum(sum(n.^2)))*ysimu;
        xsimu=ysimu+n;

        % Write WAV file (1ch track: out_channels only)
        for c=out_channels,
            audiowrite([udir uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
            audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:, c),fs);
            audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu(:, c), fs);
        end
        end
    end
    end  % ~dt05_only: end of tr05 block

    %% Create simulated development and test datasets from booth recordings %%
    sets={'dt05' 'et05'};
    for set_ind=1:length(sets),
        set=sets{set_ind};
        if strcmp(sets_filter, 'dt05_only') && ~strcmp(set, 'dt05'), continue; end

        % Read official annotations
        if official,
            mat=json2mat([apath set '_simu.json']);

        % Create new (non-official) annotations
        else
            mat=json2mat([apath set '_real.json']);
            clean_mat=json2mat([apath set '_bth.json']);
            for utt_ind=1:length(mat),
                for clean_ind=1:length(clean_mat), % match noisy utterance with same clean utterance (may be from a different speaker)
                    if strcmp(clean_mat{clean_ind}.wsj_name,mat{utt_ind}.wsj_name),
                        break;
                    end
                end
                noise_mat=mat{utt_ind};
                mat{utt_ind}=clean_mat{clean_ind};
                mat{utt_ind}.environment=noise_mat.environment;
                mat{utt_ind}.noise_wavfile=noise_mat.wavfile;
                dur=mat{utt_ind}.end-mat{utt_ind}.start;
                noise_dur=noise_mat.end-noise_mat.start;
                pbeg=round((dur-noise_dur)/2*16000)/16000;
                pend=round((dur-noise_dur)*16000)/16000-pbeg;
                mat{utt_ind}.noise_start=noise_mat.start-pbeg;
                mat{utt_ind}.noise_end=noise_mat.end+pend;
                mat{utt_ind}=orderfields(mat{utt_ind});
            end
            mat2json(mat,[apath set '_simu_new.json']);
        end

        % Loop over utterances
        if use_parfor
        parfor utt_ind=1:length(mat),
            if official,
                udir=[upath_simu set '_' lower(mat{utt_ind}.environment) '_simu/'];
                udir_ext=[upath_ext set '_' lower(mat{utt_ind}.environment) '_simu/'];
            else
                udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu_new/'];
                udir_ext=[upath_ext set '_' lower(mat{utt_ind}.environment) '_simu_new/'];
            end
            if ~exist(udir,'dir'),
                system(['mkdir -p ' udir]);
            end
            if ~exist(udir_ext,'dir'),
                system(['mkdir -p ' udir_ext]);
            end
            oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
            nname=mat{utt_ind}.noise_wavfile;
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            tbeg=round(mat{utt_ind}.noise_start*16000)+1;
            tend=round(mat{utt_ind}.noise_end*16000);

            % Load WAV files
            o=audioread([upath set '_bth/' oname '.CH0.wav']);
            [r,fs]=audioread([cpath nname '.CH0.wav'],[tbeg tend]);
            nsampl=length(r);
            x=zeros(nsampl,nchan);
            for c=1:nchan,
                ch = ref_ch*(nchan==1) + c*(nchan>1);
                x(:,c)=audioread([cpath nname '.CH' int2str(ch) '.wav'],[tbeg tend]);
            end

            % Compute the STFT (short window)
            R=stft_multi(r.',wlen_sub);
            X=stft_multi(x.',wlen_sub);

            % Estimate 88 ms impulse responses on 250 ms time blocks
            A=estimate_ir(R,X,blen_sub,ntap_sub,del);

            % Filter and subtract close-mic speech
            Y=apply_ir(A,R,del);
            y=istft_multi(Y,nsampl).';
            level=sum(sum(y.^2));
            n=x-y;

            % Compute the STFT (long window)
            O=stft_multi(o.',wlen_add);
            X=stft_multi(x.',wlen_add);
            [nbin,nfram] = size(O);

            % Localize and track the speaker
            [~,TDOAx]=localize(X,[1:nchan]);

            % Interpolate the spatial position over the duration of clean speech
            TDOA=zeros(nchan,nfram);
            for c=1:nchan,
                TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
            end

            % Filter clean speech
            Ysimu=zeros(nbin,nfram,nchan);
            for f=1:nbin,
                for t=1:nfram,
                    Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
                    Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
                end
            end
            ysimu=istft_multi(Ysimu,nsampl).';

            % Valid (dt05): random SNR in [-5,15] dB like train for matched val distribution. Test (et05): baseline at 0 dB + per-SNR [-5,0,5,10,15] dB.
            if strcmp(set,'dt05'),
                snr_baseline_dB = snr_train_range(1) + (snr_train_range(2)-snr_train_range(1))*rand();
            else
                snr_baseline_dB = 0;
            end
            SNR_linear = 10^(snr_baseline_dB/10);
            ysimu_scaled = sqrt(SNR_linear*sum(sum(n.^2))/sum(sum(ysimu.^2)))*ysimu;
            xsimu = ysimu_scaled + n;
            if ~exist(udir,'dir'), system(['mkdir -p ' udir]); end
            if ~exist(udir_ext,'dir'), system(['mkdir -p ' udir_ext]); end
            for c=out_channels,
                audiowrite([udir uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
                audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:,c),fs);
                audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu_scaled(:,c),fs);
            end
            % Test (et05) only: add per-SNR test sets [-5, 0, 5, 10, 15] dB
            if strcmp(set,'et05'),
                for snr_idx=1:length(snr_eval_list),
                    snr_dB = snr_eval_list(snr_idx);
                    SNR_linear = 10^(snr_dB/10);
                    ysimu_scaled = sqrt(SNR_linear*sum(sum(n.^2))/sum(sum(ysimu.^2)))*ysimu;
                    xsimu = ysimu_scaled + n;
                    udir_snr = [upath_simu set '_' lower(mat{utt_ind}.environment) '_simu_snr' num2str(snr_dB) '/'];
                    udir_ext_snr = [upath_ext set '_' lower(mat{utt_ind}.environment) '_simu_snr' num2str(snr_dB) '/'];
                    if ~exist(udir_snr,'dir'), system(['mkdir -p ' udir_snr]); end
                    if ~exist(udir_ext_snr,'dir'), system(['mkdir -p ' udir_ext_snr]); end
                    for c=out_channels,
                        audiowrite([udir_snr uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
                        audiowrite([udir_ext_snr uname '.CH' int2str(c) '.Noise.wav'],n(:,c),fs);
                        audiowrite([udir_ext_snr uname '.CH' int2str(c) '.Clean.wav'],ysimu_scaled(:,c),fs);
                    end
                end
            end
        end
        else
        for utt_ind=1:length(mat),
            if official,
                udir=[upath_simu set '_' lower(mat{utt_ind}.environment) '_simu/'];
                udir_ext=[upath_ext set '_' lower(mat{utt_ind}.environment) '_simu/'];
            else
                udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu_new/'];
                udir_ext=[upath_ext set '_' lower(mat{utt_ind}.environment) '_simu_new/'];
            end
            if ~exist(udir,'dir'),
                system(['mkdir -p ' udir]);
            end
            if ~exist(udir_ext,'dir'),
                system(['mkdir -p ' udir_ext]);
            end
            oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
            nname=mat{utt_ind}.noise_wavfile;
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            tbeg=round(mat{utt_ind}.noise_start*16000)+1;
            tend=round(mat{utt_ind}.noise_end*16000);

            % Load WAV files
            o=audioread([upath set '_bth/' oname '.CH0.wav']);
            [r,fs]=audioread([cpath nname '.CH0.wav'],[tbeg tend]);
            nsampl=length(r);
            x=zeros(nsampl,nchan);
            for c=1:nchan,
                ch = ref_ch*(nchan==1) + c*(nchan>1);
                x(:,c)=audioread([cpath nname '.CH' int2str(ch) '.wav'],[tbeg tend]);
            end

            % Compute the STFT (short window)
            R=stft_multi(r.',wlen_sub);
            X=stft_multi(x.',wlen_sub);

            % Estimate 88 ms impulse responses on 250 ms time blocks
            A=estimate_ir(R,X,blen_sub,ntap_sub,del);

            % Filter and subtract close-mic speech
            Y=apply_ir(A,R,del);
            y=istft_multi(Y,nsampl).';
            level=sum(sum(y.^2));
            n=x-y;

            % Compute the STFT (long window)
            O=stft_multi(o.',wlen_add);
            X=stft_multi(x.',wlen_add);
            [nbin,nfram] = size(O);

            % Localize and track the speaker
            [~,TDOAx]=localize(X,[1:nchan]);

            % Interpolate the spatial position over the duration of clean speech
            TDOA=zeros(nchan,nfram);
            for c=1:nchan,
                TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
            end

            % Filter clean speech
            Ysimu=zeros(nbin,nfram,nchan);
            for f=1:nbin,
                for t=1:nfram,
                    Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
                    Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
                end
            end
            ysimu=istft_multi(Ysimu,nsampl).';

            % Valid (dt05): random SNR in [-5,15] dB like train for matched val distribution. Test (et05): baseline at 0 dB + per-SNR [-5,0,5,10,15] dB.
            if strcmp(set,'dt05'),
                snr_baseline_dB = snr_train_range(1) + (snr_train_range(2)-snr_train_range(1))*rand();
            else
                snr_baseline_dB = 0;
            end
            SNR_linear = 10^(snr_baseline_dB/10);
            ysimu_scaled = sqrt(SNR_linear*sum(sum(n.^2))/sum(sum(ysimu.^2)))*ysimu;
            xsimu = ysimu_scaled + n;
            if ~exist(udir,'dir'), system(['mkdir -p ' udir]); end
            if ~exist(udir_ext,'dir'), system(['mkdir -p ' udir_ext]); end
            for c=out_channels,
                audiowrite([udir uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
                audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:,c),fs);
                audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu_scaled(:,c),fs);
            end
            % Test (et05) only: add per-SNR test sets [-5, 0, 5, 10, 15] dB
            if strcmp(set,'et05'),
                for snr_idx=1:length(snr_eval_list),
                    snr_dB = snr_eval_list(snr_idx);
                    SNR_linear = 10^(snr_dB/10);
                    ysimu_scaled = sqrt(SNR_linear*sum(sum(n.^2))/sum(sum(ysimu.^2)))*ysimu;
                    xsimu = ysimu_scaled + n;
                    udir_snr = [upath_simu set '_' lower(mat{utt_ind}.environment) '_simu_snr' num2str(snr_dB) '/'];
                    udir_ext_snr = [upath_ext set '_' lower(mat{utt_ind}.environment) '_simu_snr' num2str(snr_dB) '/'];
                    if ~exist(udir_snr,'dir'), system(['mkdir -p ' udir_snr]); end
                    if ~exist(udir_ext_snr,'dir'), system(['mkdir -p ' udir_ext_snr]); end
                    for c=out_channels,
                        audiowrite([udir_snr uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
                        audiowrite([udir_ext_snr uname '.CH' int2str(c) '.Noise.wav'],n(:,c),fs);
                        audiowrite([udir_ext_snr uname '.CH' int2str(c) '.Clean.wav'],ysimu_scaled(:,c),fs);
                    end
                end
            end
        end
        end
    end
    if ~isempty(p)
        delete(p);
    end
    end
