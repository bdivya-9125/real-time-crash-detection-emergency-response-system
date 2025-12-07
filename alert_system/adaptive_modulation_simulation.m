%% UEP_RS_AWGN.m 
% Unequal Error Protection simulation using Reed-Solomon codes (AWGN only) 
% - High/Medium/Low priority fields encoded with RS(255,223), RS(127,111), RS(63,55) 
% - 16-QAM modulation (bit input) 
% - BER vs SNR and Critical-field delivery probability plotted 
% Notes: Designed to avoid the RS encoder/decoder dimension errors by: 
%   * Padding user info to RS K-size 
%   * Using a consistent 8-bit expansion for TX packing 
%   * Reshaping RX bits exactly to symbol blocks prior to decoding 
%   * Applying truncate/pad safety prior to decoder calls 
clear; clc; close all; 
rng(0); % reproducible 
%% ---------------- Parameters ---------------- 
Nframes = 200;            
SNRdB = 0:2:20;           
M = 16;                   
% packets per SNR point (increase for smoother curves) 
% SNR range in dB 
% 16-QAM 
bitsPerSym = log2(M);     % 4 
% Information lengths in symbols (these are the *useful* symbols) 
lenHigh = 20;   % high-priority information symbols (each symbol ~ 8-bit for RS(255,223)) 
lenMed  = 30;   % medium-priority info symbols 
lenLow  = 40;   % low-priority info symbols 
% RS code parameters (standard sizes) 
nH = 255; kH = 223; 
nM = 127; kM = 111; 
nL = 63;  kL = 55; 
%% Validate lengths (quick sanity) 
if lenHigh > kH || lenMed > kM || lenLow > kL 
error('Information length for a field must be <= the code''s K.'); 
end 
%% Create RS encoder/decoder System objects 
encH = comm.RSEncoder(nH,kH); 
decH = comm.RSDecoder(nH,kH); 
encM = comm.RSEncoder(nM,kM); 
decM = comm.RSDecoder(nM,kM); 
encL = comm.RSEncoder(nL,kL); 
decL = comm.RSDecoder(nL,kL); 
%% Preallocate results 
BER_AWGN = zeros(size(SNRdB)); 
CritProb_AWGN = zeros(size(SNRdB)); 
%% Main simulation loop 
for si = 1:length(SNRdB) 
snr = SNRdB(si); 
totalErrBits = 0; 
totalBits = 0; 
critSuccess = 0; 
    for frame = 1:Nframes 
        %% ---------- Generate random info symbols ---------- 
        % Note: RS encoder expects integer symbols in range [0, 2^m - 1]. 
        % For RS(255,223) -> GF(2^8) -> symbols 0..255 
        msgH = randi([0, 255], lenHigh, 1);   % high - use up to 8-bit symbols 
        msgM = randi([0, 127], lenMed, 1);    % medium - up to 7-bit symbols 
        msgL = randi([0, 63],  lenLow, 1);    % low - up to 6-bit symbols 
 
        %% ---------- Pad to RS K-length ---------- 
        padH = zeros(kH - lenHigh, 1); 
        padM = zeros(kM - lenMed, 1); 
        padL = zeros(kL - lenLow, 1); 
 
        msgHpad = [msgH; padH];   % length kH 
        msgMpad = [msgM; padM];   % length kM 
        msgLpad = [msgL; padL];   % length kL 
 
        %% ---------- Encode with RS ---------- 
        % encX returns a column vector of length nX 
        codeH = encH(msgHpad);    % length nH 
        codeM = encM(msgMpad);    % length nM 
        codeL = encL(msgLpad);    % length nL 
 
        %% ---------- Convert codeword symbols to bits ---------- 
        % We force an 8-bit expansion for each symbol for uniform packing. 
        % This avoids vertical-concatenation dimension issues. 
        bitsH = de2bi(codeH,8,'left-msb');  % nH x 8 
        bitsM = de2bi(codeM,8,'left-msb');  % nM x 8 
        bitsL = de2bi(codeL,8,'left-msb');  % nL x 8 
 
        % Stack rows then linearize column-wise (consistent with reshape on RX) 
        txBitMatrix = [bitsH; bitsM; bitsL];   % (nH+nM+nL) x 8 
        txBits = txBitMatrix(:);               % column-major flattening -> vector 
 
        %% ---------- Ensure modulator bit-grouping multiple ---------- 
        if mod(length(txBits), bitsPerSym) ~= 0 
            % Pad with zeros to multiple of bitsPerSym (very unlikely here) 
            padBits = bitsPerSym - mod(length(txBits), bitsPerSym); 
            txBits = [txBits; zeros(padBits,1)]; 
        end 
 
        %% ---------- Modulate (16-QAM, bit input) ---------- 
        txSym = qammod(txBits, M, 'InputType','bit','UnitAveragePower',true); 
 
        %% ---------- AWGN Channel ---------- 
        rxSym = awgn(txSym, snr, 'measured'); 
 
        %% ---------- Demodulate ---------- 
        rxBits = qamdemod(rxSym, M, 'OutputType','bit','UnitAveragePower',true); 
 
        %% ---------- BER counting ---------- 
        L = min(length(txBits), length(rxBits)); 
        totalErrBits = totalErrBits + sum(txBits(1:L) ~= rxBits(1:L)); 
        totalBits = totalBits + L; 
 
        %% ---------- Extract and reconstruct High-priority codeword bits ---------- 
        nTotalSymbols = nH + nM + nL; 
        expectedBitsTotal = nTotalSymbols * 8; 
        if length(rxBits) < expectedBitsTotal 
            % Defensive: pad rxBits if demod returned fewer bits (shouldn't happen) 
            rxBits = [rxBits; zeros(expectedBitsTotal - length(rxBits),1)]; 
        elseif length(rxBits) > expectedBitsTotal 
            % Trim any extra bits (shouldn't happen under normal circumstances) 
            rxBits = rxBits(1:expectedBitsTotal); 
        end 
 
        % High block is the first nH symbols -> first nH*8 bits 
        rxHbits = rxBits(1 : nH*8); 
 
        % Reshape to get one row per symbol (nH x 8). 
        % IMPORTANT: we used column-major flattening at TX, so reshape with [] rows: 
        rxHmat = reshape(rxHbits, [], 8);   % yields nH x 8 matrix 
        % Convert each row (symbol bits) to integer symbol values 
        rxHsyms = bi2de(rxHmat, 'left-msb');  % nH x 1 
 
        % Defensive size-check / pad/truncate to ensure decoder gets nH symbols 
        if length(rxHsyms) < nH 
            rxHsyms = [rxHsyms; zeros(nH - length(rxHsyms),1)]; 
        elseif length(rxHsyms) > nH 
            rxHsyms = rxHsyms(1:nH); 
        end 
 
        %% ---------- RS decode high-priority ---------- 
        % decH expects a codeword-length vector (nH) and returns kH info symbols 
        % We wrap decoding in a try/catch to ensure no unhandled exception occurs, 
        % but with our size checks this should not throw. 
        try 
            recHinfo = decH(rxHsyms);   % length kH 
        catch ME 
            % In case of any RS decoder internal error, treat as failure for this frame 
            warning('RS decode error at frame %d, SNR %0.1f dB: %s', frame, snr, 
ME.message); 
            recHinfo = zeros(kH,1); 
        end 
 
        % Check the first lenHigh symbols of the decoded info against original msgH 
        if isequal(recHinfo(1:lenHigh), msgH) 
            critSuccess = critSuccess + 1; 
        end 
 
    end % frame loop 
 
    BER_AWGN(si) = totalErrBits / totalBits; 
    CritProb_AWGN(si) = critSuccess / Nframes; 
 
    fprintf('SNR = %2d dB: BER = %.3e, Critical success = %.3f\n', snr, BER_AWGN(si), 
CritProb_AWGN(si)); 
end % SNR loop 
 
%% ---------------- Plots ---------------- 
figure; 
semilogy(SNRdB, BER_AWGN, '-o', 'LineWidth', 1.8); 
grid on; xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)'); 
title('BER vs SNR (AWGN) - UEP with RS codes'); 
f
igure; 
plot(SNRdB, CritProb_AWGN, '-s', 'LineWidth', 1.8); 
grid on; xlabel('SNR (dB)'); ylabel('Critical-Field Delivery Probability'); 
title('High-priority (critical) delivery probability vs SNR'); 
%% ---------------- Done ---------------- 
disp('Simulation finished.'); 
