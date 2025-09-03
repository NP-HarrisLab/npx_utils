path = 'D:\SalineTests\23107811314';
stream = 'ap';
peakPosHz = 1000;
maxChan = 384;
overwrite = false;

runFolders = findMatchDir(path, '_g\d+$');
for i = 1:numel(runFolders)
    runname = runFolders{i};
    runDir = fullfile(path, runname);
    
    probeFolders = findMatchDir(runDir, '_imec\d+$');
    for j = 1:numel(probeFolders)
        probeDir = probeFolders{j};
        probeNum = regexp(probeDir, '_imec(\d+)$', 'tokens');
        probeNum = str2double(probeNum{1}{1});
        outputName = sprintf('%s_spectra.txt', probeDir);
        outputPath = fullfile(runDir, outputName);
        if ~overwrite && isfile(outputPath)
            continue
        end
        [ptp_mean, ptp_std, noise_mean, noise_std, h1] = ns_LFP_Power_SGLX(runDir, stream, probeNum, peakPosHz, maxChan);
        
        fout = fopen(outputPath,'w');
        fprintf('\tnoise mean & std: %.2f, %.2f\n', noise_mean, noise_std);
        fprintf(fout,'noise mean & std: %.2f, %.2f\n', noise_mean, noise_std);
        fclose(fout);
        exportgraphics(h1, fullfile(runDir, sprintf('%s_spectra.png', probeDir), 'Resolution', '300'))
    end
end

function matchFolders = findMatchDir(path, pattern)
    allContents = dir(path);
    dirFlags = [allContents.isdir];
    subFolders = allContents(dirFlags);
    subFolderNames = {subFolders.name};
    isMatch = ~cellfun('isempty', regexp(subFolderNames, pattern));
    matchFolders = subFolderNames(isMatch);
end