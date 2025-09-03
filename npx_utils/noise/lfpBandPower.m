function [lfpByChannel, allPowerEst, F, allPowerVar, outLiers, ptp_mean, ptp_std, ...
    noise_mean, noise_std] = lfpBandPower(lfpFilename, lfpFs, lfp_tomV, peakPosHz, ...
    nChansInFile, measChan, freqBand, nClips, clipDur, skipChan, ...
    bPP, noiseRange, tStart, shank_index, bank_index)
% function [lfpByChannel, allPowerEst, F] = lfpBandPower(lfpFilename, lfpFs, nChansInFile, freqBand)
% Computes the power in particular bands, and across all frequencies, across the recording
% samples 10 segments of 10 sec each to compute these things. 

if ~isempty(freqBand) && ~iscell(freqBand)
    freqBand = {freqBand};
end
nF = length(freqBand);


% load nClips one-sec samples
d = dir(lfpFilename);
nSamps = d.bytes/2/nChansInFile;

% used for spreading the clips through the whole file; not appropriate for
% survey data
%sampStarts = round(linspace(lfpFs*tStart, nSamps, nClips+1)); 

% for survey data
% start times for clips:
secStarts = tStart + (0:nClips-1)*(clipDur + 1);
sampStarts = round(secStarts*lfpFs);
nClipSamps = round(lfpFs*clipDur);

mmf = memmapfile(lfpFilename, 'Format', {'int16', [nChansInFile nSamps], 'x'});

allPowerEstByBand = zeros(nClips, nChansInFile, nF);
for n = 1:nClips
    % fprintf(1, 'clip%d\n', n);
    % pull out the data
    thisDat = lfp_tomV*(double(mmf.Data.x(:, (1:nClipSamps)+sampStarts(n))));
    % median subtract? 
%     thisDat = bsxfun(@minus, thisDat, median(thisDat));
    thisDat = bsxfun(@minus, thisDat, mean(thisDat,2));


    [Pxx, F, NFFT] = myTimePowerSpectrumMat(thisDat', lfpFs);
    
    if n==1
        allPowerEst = zeros(nClips, size(Pxx,1), size(Pxx,2));
    end
    allPowerEst(n,:,:) = Pxx;
        
    for f = 1:nF
        
        inclF = F>freqBand{f}(1) & F<=freqBand{f}(2);
        allPowerEstByBand(n,:, f) = mean(Pxx(inclF,:));
        
    end
end

if nF>0
    lfpByChannel = squeeze(mean(allPowerEstByBand, 1)); % mean across clips
else
    lfpByChannel = [];
end
allPowerVar = squeeze(var(allPowerEst,1));
allPowerEst = squeeze(mean(allPowerEst, 1));



hzSpan = ((F(2)-F(1)));
%integration window, ~1 for lfp, ~16 for ap. Replace w/ estimated width
%for welch PSD of single frequency
nHz = 16;  % 16 Hz for ap
intWind = ceil(nHz/hzSpan);
backSkipPeak = 1;
%find peaks for each channel, in a window about the expected peak
[~, pk_bin_ind] = min(abs(F-peakPosHz));
% fprintf('pk_bin_ind: %d\n', pk_bin_ind)
bStart = pk_bin_ind-10*intWind;
if bStart < 1
    bStart = 1;
end
bEnd = pk_bin_ind + 10*intWind;
% fprintf('pk_bin_ind, bStart, bEnd: %d, %d, %d\n', pk_bin_ind, bStart, bEnd);
[~,i] = max(allPowerEst(bStart:bEnd,:));
i = i + bStart;  % to index into the original array
% fprintf( 'Chan 0 peak index: %d, peak freq: %.2f Hz \n', i(1), F(i(1)));

[nChan,nPSDBin] = size(allPowerEst);

if measChan > 3

    %Loop over the estimated power spectra, integrate over the peak with a
    %window of 1 Hz (appropriate only for sine wave testing), subtracting off 
    %background estiamted from the start and end of the integration window

    peakToPeakEst = zeros(1,measChan-numel(skipChan),'single');
    backIntegrated = zeros(1,measChan-numel(skipChan),'single');
    chanLabel = 0:measChan-1;
    chanLabel(skipChan) = [];
    nPowerEst = length(allPowerEst(:,1));
    for j = 1:length(chanLabel)
        currChan = chanLabel(j)+1;
        %calc start and end windows for integrating peak 
        bStart = i(currChan)-2*intWind;
        bEnd = i(currChan)-intWind;
        if (bStart < 1), bStart = 1; end
        minPeakStart = bStart;
        if (bEnd > nPowerEst), bEnd = nPowerEst; end
        back1 = allPowerEst(bStart:bEnd,currChan);
        bStart = i(currChan)+intWind;  % i = index of max for current channel, within window about expected peak
        bEnd = i(currChan)+2*intWind;
        if (bStart < 1), bStart = 1; end
        if (bEnd > nPowerEst), bEnd = nPowerEst; end
        maxPeakEnd = bEnd;
        back2 = allPowerEst(bStart:bEnd,currChan);
        backBoth = cat(1,back1,back2);
        backEst = mean(backBoth);
        peakInt = 0;
        bStart = i(currChan)-intWind;
        bEnd = i(currChan)+intWind;
        if (bStart < 1) bStart = 1; end
        if (bEnd > nPowerEst) bEnd = nPowerEst; end
        for f = bStart:bEnd
            peakInt = peakInt + allPowerEst(f,currChan) - backEst;
        end
        if peakInt > 0
            peakToPeakEst(j) = 2*sqrt(2*peakInt/((1/lfpFs)*NFFT));
        end
        %integrate over specified range for a noise est

        bStart = round(noiseRange(1)/hzSpan)+1;
        bEnd = round(noiseRange(2)/hzSpan)+1;
        backSum = 0;
        msgStr = sprintf( 'minPeakStart, maxPeakEnd: %d, %d\n', minPeakStart, maxPeakEnd );
        %disp(msgStr)
        if (backSkipPeak == 1)
            if( bStart < minPeakStart ) && ( bEnd > maxPeakEnd )
                backSum = sum(allPowerEst(bStart:minPeakStart,currChan)) + ...
                            sum(allPowerEst(maxPeakEnd:bEnd,currChan)); 
            
            elseif(( bStart < minPeakStart ) && ( bEnd <= minPeakStart )) ...
                || (( bStart >= maxPeakEnd ) && ( bEnd > maxPeakEnd ))  
                %fprintf('bStart, bEnd: %d, %d\n', bStart,bEnd);

                backSum = sum(allPowerEst(bStart:bEnd,currChan)); 
            elseif( bStart < minPeakStart ) && ( bEnd <= maxPeakEnd )
                backSum = sum(allPowerEst(bStart:minPeakStart,currChan)); 
            elseif( bStart >= minPeakStart ) && ( bEnd > maxPeakEnd )
                backSum = sum(allPowerEst(maxPeakEnd,currChan)); 
            else
                backSum = 0;
            end  
                      
        else
           if( currChan == 1) 
            [pxx_nf, pxx_nchan] = size(allPowerEst);
            fprintf('number of bins in power spectrum: %d\n ', pxx_nf);
            fprintf('range for backSum: %d, %d\n ', bStart, bEnd);

           end
           backSum = sum(allPowerEst(bStart:bEnd,currChan));

        end
        if backSum > 0
            %Note this is an rms estimate, in contrast to the peak to peak
            %estimate used for understanding the sine wave.
            backIntegrated(j) = sqrt(backSum/((1/lfpFs)*NFFT));
        end
        
    end
    
    

    squeeze(peakToPeakEst);
    squeeze(backIntegrated);
    
    %write out peakToPeakEst and chanLabel to command window
%     for i = 1:length(chanLabel)
%         fprintf( '%d\t%.3f\n', chanLabel(i), peakToPeakEst(i));
%     end
    
    normPP = peakToPeakEst/mean(peakToPeakEst);
    [filePath, currTitle, dumExt] = fileparts(lfpFilename);
    currTitle = sprintf('%s_sh%d_b%d_t%d', currTitle, shank_index, bank_index, round(tStart,0));
    titleStr = sprintf( 'Data from: %s',currTitle);

    figure(shank_index*4 + bank_index + 1);
    subplot(1,2,1);
    scatter(chanLabel,peakToPeakEst)
    title(titleStr,'Interpreter', 'none');
    xlabel('channel')
    ylabel('peak to peak (mV)');
    
    subplot(1,2,2);
    edges = (0:0.01:1.5);
    histogram(normPP, edges)
    xlabel('normalized p-p')
    ylabel('number of channels');
    
    %plot skipped channels + neighbors
%     figure6 = figure;
%     for j = 1:length(skipChan)
%         currChan = skipChan(j)+1;
%         plot(F,(allPowerEst(:,currChan-1)),F,(allPowerEst(:,currChan)),F,(allPowerEst(:,currChan+1))  );
%         hold on
%     end
%     title('skipped and neighbor chan PSDs');
    
    figure(shank_index*4 + bank_index + 101);
    scatter(chanLabel,backIntegrated)
    titleStr2 = sprintf( 'Data from: %s\nIntegrating %.1f-%.1f',currTitle, noiseRange(1), noiseRange(2));
    title(titleStr2,'Interpreter', 'none');
    xlabel('channel')
    ylabel('background(mV)');
    
    %write out a simple text file of the estimated powers and noise
    outDat = zeros(length(chanLabel),3);
    outDat(:,1) = chanLabel;
    outDat(:,2) = 1000*peakToPeakEst;
    outDat(:,3) = 1000*backIntegrated;
    out_table = array2table(outDat,...
    'VariableNames',{'chan','Vpp_est_uV','back_int_uV'});
    newFile = sprintf( '%s.txt', currTitle);
    writetable(out_table, fullfile(filePath, newFile));
    % fprintf('mean_ptp, std_ptp: %.2f, %.2f\n', mean(outDat(:,2)), std(outDat(:,2)))
    % fprintf('mean_integrated noise, std: %.2f, %.2f\n', mean(outDat(:,3)), std(outDat(:,3)))
    ptp_mean = mean(outDat(:,2));
    ptp_std = std(outDat(:,2));
    noise_mean = mean(outDat(:,3));
    noise_std = std(outDat(:,3));
    if (bPP == 0)
        [~, backIndex] = sort(backIntegrated);
        outLiers=[ (backIndex(1:3)), (backIndex(end-2:end)) ];
    else
       [~, ppIndex] = sort(peakToPeakEst);
        outLiers=[ (ppIndex(1:3)), (ppIndex(end-2:end)) ];
    end 
else

   back1 = allPowerEst(i-2*intWind:i-intWind);
   back2 = allPowerEst(i+intWind:i+2*intWind);
   backBoth = cat(2,back1,back2);
   backEst = mean(backBoth);
   peakInt = 0;
   for f = i-intWind:i+intWind
       peakInt = peakInt + allPowerEst(f) - backEst;
   end

   peakToPeakEst = 2*sqrt(2*peakInt/((1/lfpFs)*NFFT));
    ptp_mean = peakToPeakEst;
    ptp_std = 0;
    noise_mean = 0;
    noise_std =0;

   titleStr = sprintf('Welch power spectrum estimate, p-p est = %0.2f mV',peakToPeakEst);
   %make a simple plot of the power spectrum
   plot( F, 10*log10(allPowerEst) );
   title(titleStr);
   xlabel('frequency (Hz)');
   ylabel('power (dB)');
   outLiers = 1;
end


function [Pxx, F, NFFT] = myTimePowerSpectrumMat(x, Fs)
L = size(x,1);
NFFT = 2^nextpow2(L);
[Pxx,F] = pwelch(x,[],[],NFFT,Fs);