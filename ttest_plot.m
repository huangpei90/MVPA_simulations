function ttest_plot()
y=load('results.mat');
results = y.results;
design_names = fieldnames(results(1));
MVPA_names = fieldnames(results(1).(design_names{1}));
CNR = y.params.CNR;

for i=1:size(design_names,1)
    for j=1:size(MVPA_names,1)
        results_mat = zeros(size(results(1).(design_names{1}),2));
        for iter=1:size(results,2)
            for m=1:size(results(1).(design_names{1}),2)
                for n=1:size(results(1).(design_names{1}),2)
                    results_mat(m,n) = results_mat(m,n) + ttest([results(iter).(design_names{i})(:,m).(MVPA_names{j})],[results(iter).(design_names{i})(:,n).(MVPA_names{j})]);
                end
            end
        end
        results_mat =results_mat/size(results,2);
        plot.(design_names{i}).(MVPA_names{j}) = results_mat;
    end
end
    
if ~exist('graphs', 'dir')
    mkdir('graphs')
end

for i=1:size(design_names,1)
    for j=1:size(MVPA_names,1)    
        figure
        imagesc(plot.(design_names{i}).(MVPA_names{j}))
        fname = sprintf('graphs/%s_%s.png',design_names{i},MVPA_names{j});
        set(gca,'xtick',1:5:size(CNR,2),'xticklabel',CNR(1:5:end));
        set(gca,'ytick',1:5:size(CNR,2),'yticklabel',CNR(1:5:end));
        saveas(gcf,fname,'png');
        close all
    end
end
end