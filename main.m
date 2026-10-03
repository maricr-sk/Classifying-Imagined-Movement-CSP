clear; clc; close all;

eeglabroot = '/Users/marissacrevecoeur/Documents/MATLAB/eeglab14_1_2b';
addpath(genpath(eeglabroot)); rehash;

trigger_ch = 18;
eeg_ch     = 1:16;
Fs = 256;

[b,a] = butter(4, [7 13]/(Fs/2), 'bandpass');

files = {
    'MotorImageryCSP_Data_run1_session2.mat'
    'MotorImageryCSP_Data_run2_session2.mat'
    'MotorImageryCSP_Data_run3_session2.mat'
    'MotorImageryCSP_Data_run4_session2.mat'
};

all_left  = {};
all_right = {};

win_start = round(0.5 * Fs);
win_end   = round(2.5 * Fs);

for f = 1:length(files)
    S = load(files{f});
    Y = S.y;

    EEG  = Y(eeg_ch, :);
    trig = Y(trigger_ch, :);

    EEG = filtfilt(b, a, EEG')';

    thr = 0.5 * max(trig);
    trig_bin = trig > thr;
    onset_idx = find(diff(trig_bin) == 1) + 1;

    if isempty(onset_idx), continue; end

    amps = trig(onset_idx);

    lab = kmeans(amps(:), 2, 'Replicates', 10);

    m1 = mean(amps(lab==1));
    m2 = mean(amps(lab==2));
    if m1 < m2
        left_lab = 1; right_lab = 2;
    else
        left_lab = 2; right_lab = 1;
    end

    for t = 1:numel(onset_idx)
        idx0 = onset_idx(t);
        idx_start = idx0 + win_start;
        idx_end   = idx0 + win_end;

        if idx_start < 1 || idx_end > size(EEG,2), continue; end

        seg = EEG(:, idx_start:idx_end);

        if lab(t) == left_lab
            all_left{end+1} = seg;
        elseif lab(t) == right_lab
            all_right{end+1} = seg;
        end
    end
end

fprintf('Collected %d left trials and %d right trials.\n', numel(all_left), numel(all_right));
assert(~isempty(all_left) && ~isempty(all_right), 'No trials collected. Check trigger labeling/windowing.');

X_left  = cat(3, all_left{:});
X_right = cat(3, all_right{:});

nKeep = min(size(X_left,3), size(X_right,3));
idxL = randperm(size(X_left,3), nKeep);
idxR = randperm(size(X_right,3), nKeep);
X_left  = X_left(:,:,idxL);
X_right = X_right(:,:,idxR);

[nCh, nSamp, nL] = size(X_left);
[~, ~, nR] = size(X_right);
fprintf('Balanced to %d trials per class.\n', nKeep);

cov_func = @(X) (X*X') / trace(X*X');

C_L = zeros(nCh);
for i = 1:nL, C_L = C_L + cov_func(X_left(:,:,i)); end
C_L = C_L / nL;

C_R = zeros(nCh);
for i = 1:nR, C_R = C_R + cov_func(X_right(:,:,i)); end
C_R = C_R / nR;

eps_reg = 1e-6;
C_L = C_L + eps_reg * eye(nCh) * trace(C_L)/nCh;
C_R = C_R + eps_reg * eye(nCh) * trace(C_R)/nCh;

[W, D] = eig(C_L, C_R);
[eigvals, idx] = sort(diag(D), 'ascend');
W = W(:, idx);

W_csp = [W(:,1:3), W(:,end-2:end)];
nComp = size(W_csp,2);

ZL = zeros(nComp, nSamp, nL);
ZR = zeros(nComp, nSamp, nR);
for i = 1:nL, ZL(:,:,i) = W_csp' * X_left(:,:,i); end
for i = 1:nR, ZR(:,:,i) = W_csp' * X_right(:,:,i); end

t = (0:nSamp-1) / Fs;

meanL = mean(ZL, 3);
meanR = mean(ZR, 3);

featL = squeeze(std(ZL,0,2))';   % nL x nComp
featR = squeeze(std(ZR,0,2))';   % nR x nComp
Xfeat = [featL; featR];
y     = [ones(size(featL,1),1); 2*ones(size(featR,1),1)];

K = min(10, numel(y));
cv = cvpartition(y,'KFold',K);

acc = zeros(cv.NumTestSets,1);
for r = 1:cv.NumTestSets
    tr = training(cv,r);
    te = test(cv,r);
    mdl  = fitcdiscr(Xfeat(tr,:), y(tr), 'DiscrimType','linear');
    yhat = predict(mdl, Xfeat(te,:));
    acc(r) = mean(yhat == y(te));
end

fprintf('%d-fold accuracy: %.2f%% ± %.2f%% (SE)\n', ...
    cv.NumTestSets, 100*mean(acc), 100*std(acc)/sqrt(numel(acc)));


figure;
for c = 1:nComp
    subplot(2,3,c);
    plot(t, meanL(c,:), 'b', 'LineWidth', 1); hold on;
    plot(t, meanR(c,:), 'r', 'LineWidth', 1);
    xlim([t(1) t(end)]);
    xlabel('Time (s)');
    ylabel('Projected amplitude');
    title(sprintf('CSP dim %d', c));
    legend({'Left','Right'});
end

ch1 = 1; ch2 = 2;
beforeL = [ squeeze(std(X_left(ch1,:,:),0,2)) , squeeze(std(X_left(ch2,:,:),0,2)) ];
beforeR = [ squeeze(std(X_right(ch1,:,:),0,2)) , squeeze(std(X_right(ch2,:,:),0,2)) ];

afterL = [ squeeze(std(ZL(1,:,:),0,2)) , squeeze(std(ZL(2,:,:),0,2)) ];
afterR = [ squeeze(std(ZR(1,:,:),0,2)) , squeeze(std(ZR(2,:,:),0,2)) ];

figure;
subplot(1,2,1);
scatter(beforeL(:,1), beforeL(:,2), 40, 'b', 'filled'); hold on;
scatter(beforeR(:,1), beforeR(:,2), 40, 'r', 'filled');
xlabel(sprintf('Std, channel %d', ch1));
ylabel(sprintf('Std, channel %d', ch2));
title('Before');
legend({'Left','Right'}); grid on;

subplot(1,2,2);
scatter(afterL(:,1), afterL(:,2), 40, 'b', 'filled'); hold on;
scatter(afterR(:,1), afterR(:,2), 40, 'r', 'filled');
xlabel('Std CSP component 1');
ylabel('Std CSP component 2');
title('After');
legend({'Left','Right'}); grid on;

sgtitle('Class separability before vs after CSP');

locFile = fullfile(eeglabroot, 'BCI.locs');
assert(exist(locFile,'file')==2, 'Cannot find BCI.locs at: %s', locFile);
allLocs = readlocs(locFile, 'filetype', 'loc');

A = pinv(W)';
A_csp = A(:, [1:3, end-2:end]);

figure;
for c = 1:size(A_csp,2)
    subplot(2,3,c);
    topoplot(A_csp(:,c), allLocs(1:16));
    colorbar;
    title(sprintf('CSP pattern %d', c));
end
