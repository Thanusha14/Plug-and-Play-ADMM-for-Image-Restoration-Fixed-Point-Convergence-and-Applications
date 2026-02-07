import numpy as np
from skimage import io, img_as_float32, color
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
from bm3d import bm3d as bm3d_denoise



def to_float01(img):

    img = img_as_float32(img)
    img = np.clip(img, 0, 1)

    if img.ndim == 2:

        img = img[..., None]

    return img



def show_pair(inp, out, title_in = 'Input', title_out = 'Restored'):

    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)

    if inp.shape[-1] == 1:

        plt.imshow(inp[...,0], cmap = 'gray')

    else:

        plt.imshow(inp)

    plt.axis('off'); plt.title(title_in)
    plt.subplot(1,2,2)

    if out.shape[-1] == 1:

        plt.imshow(out[...,0], cmap='gray')

    else:

        plt.imshow(out)


    plt.axis('off'); plt.title(title_out)
    plt.show()



def denoise_tv(img, weight = 0.1):

    if img.ndim == 2 or img.shape[-1] == 1:

        arr = denoise_tv_chambolle( img[...,0] 
                                   if img.shape[-1] == 1 
                                   else img,weight=weight, channel_axis=None)

        return arr[..., None].astype(np.float32) if img.shape[-1] == 1 else arr.astype(np.float32)
    
    else:

        out = np.stack( [ denoise_tv_chambolle(img[...,c], weight=weight, channel_axis=None)
                        for c in range( img.shape[-1] ) ] , axis = -1)
        
        return out.astype(np.float32)



def denoise_nlm(img, est_sigma = 0.02):
    h = 0.8 * float(est_sigma)

    if img.shape[-1] == 1:

        return denoise_nl_means(img[...,0], h = h, fast_mode = True,
                                patch_size = 5, patch_distance = 6,
                                channel_axis =None)[...,None].astype(np.float32)
    else:

        return denoise_nl_means(img, h = h, fast_mode = True,
                                patch_size = 5, patch_distance = 6,
                                channel_axis = -1).astype(np.float32)



def denoise_bm3d(img, sigma = 0.02):

    if img.shape[-1] == 1:
        return bm3d_denoise(img[...,0], sigma_psd = sigma)[...,None].astype(np.float32)
    
    else:
        out = np.stack([bm3d_denoise(img[...,c], sigma_psd = sigma) for c in range(img.shape[-1])], axis = -1)
        return out.astype(np.float32)



def gaussian_kernel(size, sigma):

    if size % 2 == 0:
        size += 1

    ax = np.arange(-(size // 2), size//2+1, dtype = np.float32)
    xx, yy = np.meshgrid(ax, ax, indexing  ='xy')
    k = np.exp(-( xx**2 + yy**2 ) / ( 2*sigma**2 ) )
    k /= k.sum()

    return k.astype(np.float32)



def pad_psf(psf, H, W):

    ph, pw = psf.shape
    out = np.zeros((H,W), dtype=np.float32)
    out[:ph,:pw] = psf
    out = np.roll(out, -ph//2, axis=0)
    out = np.roll(out, -pw//2, axis=1)

    return out



def pnp_admm(y, task='denoise', denoiser='tv', noise_sigma=0.02, iters=30, rho0=0.1, blur_sigma=1.4, kernel_size=21):

    y = to_float01(y)
    Hh,Ww,_ = y.shape
    x = y.copy(); v = x.copy(); u = np.zeros_like(x, dtype=np.float32)
    rho = rho0

    use_deblur = (task=='deblur')
    
    if use_deblur:

        psf = gaussian_kernel(kernel_size, blur_sigma)
        psf_pad = pad_psf(psf, Hh, Ww)
        H_fft = np.fft.fft2(psf_pad)
        H_conj = np.conj(H_fft)
        denom_base = (np.abs(H_fft)**2) / (noise_sigma**2 + 1e-12)

    for k in range(iters):

        z = v - u
        x_old = x.copy()

        if task=='denoise':

            inv = 1.0 / (1.0/(noise_sigma**2 + 1e-12) + rho)
            x = (y / (noise_sigma**2 + 1e-12) + rho * z) * inv


        else:

            for c in range(x.shape[-1]):

                Zc = np.fft.fft2(z[...,c])
                Yc = np.fft.fft2(y[...,c])
                num = H_conj * Yc / (noise_sigma**2 + 1e-12) + rho * Zc
                den = denom_base + rho
                Xc = num / den
                x[...,c] = np.real(np.fft.ifft2(Xc)).astype(np.float32)

        xin = x + u

        if denoiser=='tv':
            v = denoise_tv(xin, weight=0.08)

        elif denoiser=='nlm':
            est = estimate_sigma(color.rgb2gray(y) if y.shape[-1]>1 else y[...,0], channel_axis=None)
            v = denoise_nlm(xin, est_sigma=est)

        elif denoiser=='bm3d':
            est = estimate_sigma(color.rgb2gray(y) if y.shape[-1]>1 else y[...,0], channel_axis=None)
            v = denoise_bm3d(xin, sigma=est)
            
        else:
            raise ValueError("Unknown denoiser!")

        u = u + x - v

        if (k+1) % 10 == 0: 
            rho *= 1.2

        rel = np.linalg.norm((x - x_old).ravel()) / (np.linalg.norm(x_old.ravel()) + 1e-12)

        print(f'Iter {k+1:03d} | rho = {rho:.4f} | rel = {rel:.3e}')

        if (k+1) % 5 == 0:
            print(f" Progress: {k+1}/{iters} iterations done for {denoiser.upper()}...")

        if k>5 and rel < 1e-3: 
            break

    print(f"{denoiser} finished all {iters} iterations.")


    return np.clip(x,0,1)



def psnr(img1, img2, data_range=1.0):
    
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:

        return float('inf')
    
    return 10 * np.log10((data_range ** 2) / mse)



fname = r"C:\Users\Meera\OneDrive\Documents\SEM 3\MFC\cameraman.jpeg" 
img = io.imread(fname)
imgf = to_float01(img)

print("Loaded:", fname, "shape=", imgf.shape)
plt.imshow(imgf)
plt.axis('off')
plt.title('Input')
plt.show()


#adding noise to the input data, cuz the psnr was bs. 
noise_sigma = 0.05  
noisy_img = imgf + np.random.normal(0, noise_sigma, imgf.shape).astype(np.float32)
noisy_img = np.clip(noisy_img, 0, 1)


restored_tv = pnp_admm(noisy_img, denoiser='tv', noise_sigma=noise_sigma)
restored_nlm = pnp_admm(noisy_img, denoiser='nlm', noise_sigma=noise_sigma)
restored_bm3d = pnp_admm(noisy_img, denoiser='bm3d', noise_sigma=noise_sigma)


show_pair(noisy_img, restored_tv, title_in='Noisy', title_out='Restored (TV)')
show_pair(noisy_img, restored_nlm, title_in='Noisy', title_out='Restored (NLM)')
show_pair(noisy_img, restored_bm3d, title_in='Noisy', title_out='Restored (BM3D)')


psnr_tv = psnr(imgf, restored_tv)
psnr_nlm = psnr(imgf, restored_nlm)
psnr_bm3d = psnr(imgf, restored_bm3d)

print(f"PSNR (TV)   : {psnr_tv:.2f} dB")
print(f"PSNR (NLM)  : {psnr_nlm:.2f} dB")
print(f"PSNR (BM3D) : {psnr_bm3d:.2f} dB")
