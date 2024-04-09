[file, path] = uigetfile; % don't forget to set to look for all files; find the h5 file
fpath = strcat(path, file);
info = h5info(fpath);
dsets_names = {info.Datasets.Name};
data = struct();
for i=1:numel(dsets_names)
    field_name=dsets_names{i};
    value=h5read(fpath, strcat("/",field_name)); % datasets are called /dset etc., i.e. with slash 
    data.(field_name)=value;
end