function add_path()


addpath('../../python/SUNet/');
if count(py.sys.path,'../../python/SUNet/') == 0
    insert(py.sys.path,int32(0),'../../python/SUNet/');
end
py.importlib.reload(py.importlib.import_module('denoiser_SUNet'));
end


