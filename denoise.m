function [ denoised ] = denoise(noisy,sigma_hat,width,height,denoiser)

global global_time
noisy=reshape(noisy,[width,height]);

switch denoiser
    
    case 'MWDNN'
        noisy = reshape(noisy,height,width);
        noisy = noisy';
        noisy = reshape(noisy,1,height*width);
        output = double(py.denoiser_SUNet.denoise124(noisy,sigma_hat));
        output = reshape(output,height,width);
        output = output'; 
    
    otherwise
        error('Unrecognized Denoiser')
end
denoised=output(:);
end