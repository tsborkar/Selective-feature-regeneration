f=load('ILSVRC_val.mat');
% organize validation set data
filenames = f.filename;
class_labels = f.class_label;
for class_id = 0:999
    [r,c,~]=find(class_labels==class_id);
    if(isempty(r)== 0)
        dsp_str = sprintf('\n Processing class label %d with %d files',class_id, length(r));
        display(dsp_str)
        mkdir(strcat('../ILSVRC_data/Class_',num2str(class_id)));
        out_string ='../ILSVRC_data/';
        for idx = 1:length(r)
            fl_str = sprintf('\n Filename : %s',filenames{r(idx)});
            display(fl_str)
            src_loc = strcat(out_string,filenames{r(idx)});
            dst_loc = strcat('../ILSVRC_data/Class_',num2str(class_id),'/',filenames{r(idx)});
            movefile(src_loc,dst_loc,'f');
 
        end

    end
end

clear f filenames class_labels;
% organize training set data
f=load('ILSVRC_train.mat');
filenames = f.filename;
class_labels = f.class_label;
for class_id = 0:999
    [r,c,~]=find(class_labels==class_id);
    if(isempty(r)== 0)
        dsp_str = sprintf('\n Processing class label %d with %d files',class_id, length(r));
        display(dsp_str)
        mkdir(strcat('../Training/Class_',num2str(class_id)));
        out_string ='../Training/';
        for idx = 1:length(r)
            fl_str = sprintf('\n Filename : %s',filenames{r(idx)});
            display(fl_str)
            src_loc = strcat(out_string,filenames{r(idx)});
            dst_loc = strcat('../Training/Class_',num2str(class_id),'/',filenames{r(idx)});
            movefile(src_loc,dst_loc,'f');
 
        end

    end
end

clear f filenames class_labels;

display('Finished moving data')
