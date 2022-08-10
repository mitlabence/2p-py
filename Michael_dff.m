% DFOF with moving baseline %

dfof = data - data; % data is (n_components, n_frames) matrix?
winsize = 300; % putinwinsize HERE
for c=1:size(data, 1);  % component index
    for t=winsize+1:size(data, 2);
        srted = sort(data(c, t - winsize:t - 1));
        dfof(c, t) = (data(c, t) - median(srted(1:winsize / 6))) / median(srted(1: winsize / 2));
        % dfof(c, t) = (data(c, t) - mean(data(c, t - winsize:t-1)));
        if dfof(c, t) < 0;
            
        end
    
    end
% dfof(c,:)=gradient(data(c,:));
% for t=1:size(data, 2);
% if dfof(c, t) < 0;
% dfof(c, t) = 0;
% end
% end

end
figure;
plot(dfof(:,:)'); axis([1 size(dfof,2) 0 10]); m=median(dfof,2); st=std(dfof,0,2); ii=iqr(dfof,2);
figure;
imagesc(dfof);
% m = m - m;
% st = (st - st) + 1;