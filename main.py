# coding=utf-8
import argparse, os, glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Compose, ToTensor
from torchvision.utils import save_image
from dataset import DatasetFromFolder
import numpy as np
from PIL import Image

from module import *
from module import ssim
from util import *
from ac.torchac import *
from ac import arithmeticcoding
from criterion import *

import warnings
warnings.filterwarnings("ignore")


def train(epoch):
    # Loss:
    criterion_mse = nn.MSELoss()
    criterion_msssim = ssim.MS_SSIM(nonnegative_ssim=True)
    # Set module training
    encode_model.train()
    decode_model.train()
    hyper_encode_model.train()
    hyper_decode_model.train()
    auto_regressive_model.train()
    quant_noise.train()
    quant_ste.train()
    prob_model.train()
    # Tables
    num_pixels = opt.patchSize ** 2
    train_size = len(training_data_loader)
    
    # Training iteration
    for iteration, batch in enumerate(training_data_loader, 1):
        # batch
        image, _ = batch
        image = image.to(device)
        n, c, h, w = image.shape

        # Updata lr
        lr_scheduler.update_lr(batch_size=n)
        current_lr = lr_scheduler.get_lr()
        for param_group in base_optimizer.param_groups:
            param_group['lr'] = current_lr / 3
        for param_group in entropy_optimizer.param_groups:
            param_group['lr'] = current_lr
  
        # Encoder
        y = encode_model(image)
        y_tilde = quant_noise(y)
        y_tilde2 = quant_ste(y)   # quant_ste(y), y_tilde  # modified
        # hyper predict 
        z = hyper_encode_model(y)
        z_tilde = quant_noise(z)
        z_feature = hyper_decode_model(z_tilde)
        # Regressive model
        para1, para2, para3, S, U, R = auto_regressive_model(y_tilde, z_feature, criterion_gauss)
        # Decoder
        x_tilde = decode_model(y_tilde2)
        x_tilde = torch.clamp(x_tilde, 0., 1.)
        
        # Distortion Loss 
        loss_mse = criterion_mse(x_tilde, image) * 255 * 255
        loss_ms_ssim = 1 - criterion_msssim(x_tilde*255., image*255.)
        if opt.loss_type=='mse':
            loss_distortion = loss_mse
        elif opt.loss_type=='msssim':
            loss_distortion = loss_ms_ssim
        else:
            raise ValueError("No such loss type")
            
        # Calculate bpp of z & y
        z_prob = prob_model(z_tilde)
        bpp_z = - torch.log2(z_prob + 1e-10).sum() / num_pixels / opt.batchSize
        loss_rate_z = bpp_z

        # Parameter -> Merge -> probability
        ny, cy, hy, wy = y.shape
        para_merge = torch.cat([para1.reshape(ny, opt.num_parameter, cy, 1, hy, wy),
                                para2.reshape(ny, opt.num_parameter, cy, 1, hy, wy),
                                para3.reshape(ny, opt.num_parameter, cy, 1, hy, wy)],
                                dim=3)
        para_merge = para_merge.reshape(ny, -1, hy, wy)
        # loss_rate_y = 0.7 * criterion_gauss(y_tilde, para_merge).sum() + \
        #               0.1 * criterion_gauss(y_tilde, para1).sum() + \
        #               0.1 * criterion_gauss(y_tilde, para2).sum() + \
        #               0.1 * criterion_gauss(y_tilde, para3).sum()   # modified
        loss_rate_y = criterion_gauss(y_tilde, para_merge).sum()
        loss_rate_y = loss_rate_y / np.log(2) / num_pixels / opt.batchSize
        total_loss = loss_distortion * opt.alpha + loss_rate_y + loss_rate_z
        
        # Zero Gradient
        base_optimizer.zero_grad()
        entropy_optimizer.zero_grad()
        # Backward and Update modules
        total_loss.backward()
        base_optimizer.step()
        entropy_optimizer.step()
        
        if(iteration%10 == 0):
            print_fmt = "- Epoch[{}]({}/{}) - MSE:{:.2f}, MS-SSIM:{:.4f}, bpp y:{:.3f}, bpp z:{:.3f}, lr:{:.1e}"
            log.logger.info(print_fmt.format(epoch, iteration, train_size, 
                                   loss_mse.item(), 1-loss_ms_ssim.item(),
                                   loss_rate_y.item(), bpp_z.item(), 
                                   base_optimizer.param_groups[0]['lr']))
    log.logger.info("--- Epoch {} Complete.".format(epoch))

def test(epoch=0, shape_num=64):
    criterion_mse = nn.MSELoss()
    criterion_msssim = ssim.MS_SSIM()
        
    # Set module testing
    encode_model.eval()
    decode_model.eval()
    hyper_encode_model.eval()
    hyper_decode_model.eval()
    auto_regressive_model.eval()
    quant_noise.eval()
    prob_model.eval()

    results = np.zeros((len(testing_data_loader), 6))
    bpp_z_list = []

    with torch.no_grad():
        for iteration, sample in enumerate(testing_data_loader, 1):
            # iter1 = testing_data_loader.__iter__()
            # sample = iter1.next()
        
            image, img_path = sample
            img_name = img_path[0].split('/')[-1].split('.png')[0]
            image = image.to(device)
            n, c, h, w = image.shape
            num_pixels = h * w
            
            # image padding
            image_padded = img_pad(image, shape_num)
            # Encoder
            y = encode_model(image_padded)
            # Quantization and Dequantization
            y_hat = quant_noise(y)
            # hyper predict 
            z = hyper_encode_model(y)
            z_hat = quant_noise(z)
            z_feature = hyper_decode_model(z_hat)
            # Regressive model, modified
            para1, para2, para3, S, U, R = auto_regressive_model(y_hat, z_feature, criterion_gauss)
            # Decoder
            x_hat = decode_model(y_hat)
            x_hat = torch.clamp(x_hat, 0., 1.)
            
            # image de-pad
            pad_up = ((shape_num - h % shape_num) % shape_num ) // 2
            pad_left = ((shape_num - w % shape_num) % shape_num ) // 2
            x_hat = x_hat[:, :, pad_up:pad_up+h, pad_left:pad_left+w]

            # Parameter -> Merge -> probability
            ny, cy, hy, wy = y.shape
            para_merge = torch.cat([para1.reshape(ny, opt.num_parameter, cy, 1, hy, wy),
                                    para2.reshape(ny, opt.num_parameter, cy, 1, hy, wy),
                                    para3.reshape(ny, opt.num_parameter, cy, 1, hy, wy)],
                                    dim=3)
            para_merge = para_merge.reshape(ny, -1, hy, wy)
            y_predicted_logits = criterion_gauss(y_hat, para_merge)
            y_prob = (- y_predicted_logits).exp_()
            # Calculate bpp of z
            z_prob = prob_model(z_hat)
            
            mse = criterion_mse(x_hat*255., image*255.)
            psnr = 20. * np.log10(255.) - 10 * np.log10(mse.item())
            msssim = criterion_msssim(image*255, x_hat*255)
            bpp_z = - torch.log2(z_prob + 1e-10).sum().item() / num_pixels
            bpp_y = - torch.log2(y_prob).sum().item() / num_pixels
            log.logger.info("%s - PSNR:%.2f, MS-SSIM:%.5f, bpp:%.4f/%.4f"%(img_name,psnr,msssim,bpp_y,bpp_z))
    
            results[iteration-1] = [mse.item()*num_pixels*3, psnr, msssim, h, w, (bpp_y+bpp_z)*num_pixels]
            bpp_z_list.append(bpp_z)

    npixels_ = np.multiply(results[:,3], results[:,4])
    length_ = results[:, 5].sum()
    mse_ = results[:, 0].sum() / npixels_.sum() / 3
    psnr_, msssim_ = results[:,1].mean(), results[:,2].mean()
    bpp_ = length_ / npixels_.sum()
    format_print = "* Avg. PSNR:%.2f, MS-SSIM:%.5f, bpp:%.4f, bpp z:%.4f" % (psnr_,msssim_,bpp_,np.mean(bpp_z_list))
    log.logger.info(format_print)
  
def compress(shape_num=64):
    if not os.path.exists('./compressed'):
        os.mkdir('./compressed')

    criterion_mse = nn.MSELoss()
    criterion_msssim = ssim.MS_SSIM()
    
    # Set module testing
    encode_model.eval()
    decode_model.eval()
    hyper_encode_model.eval()
    hyper_decode_model.eval()
    auto_regressive_model.eval()
    quant_noise.eval()
    prob_model.eval()
    
    with torch.no_grad():
        image = Image.open(opt.input_file).convert('RGB')
        image = ToTensor()(image).unsqueeze(0)

        image = image.to(device)
        img_name = opt.input_file.split('/')[-1].split('.png')[0]
        n, c, h, w = image.shape
        num_pixels = h * w

        # image padding
        image_padded = img_pad(image, shape_num)
        # Encoder
        y = encode_model(image_padded)
        # Quantization and Dequantization
        y_hat = quant_noise(y)
        # hyper predict 
        z = hyper_encode_model(y)
        z_hat = quant_noise(z)
        z_feature = hyper_decode_model(z_hat)
        # Regressive model, modified
        para1, para2, para3, S, U, R = auto_regressive_model(y_hat, z_feature, criterion_gauss)
        # Decoder
        x_hat = decode_model(y_hat)
        x_hat = torch.clamp(x_hat, 0., 1.)
        # image de-pad
        pad_up = ((shape_num - h % shape_num) % shape_num ) // 2
        pad_left = ((shape_num - w % shape_num) % shape_num ) // 2
        x_hat = x_hat[:, :, pad_up:pad_up+h, pad_left:pad_left+w]
        # Parameter -> Merge -> probability
        ny, cy, hy, wy = y.shape
        para_merge = torch.cat([para1.reshape(ny, opt.num_parameter, cy, 1, hy, wy),
                                para2.reshape(ny, opt.num_parameter, cy, 1, hy, wy),
                                para3.reshape(ny, opt.num_parameter, cy, 1, hy, wy)],
                                dim=3)
        para_merge = para_merge.reshape(ny, -1, hy, wy)
        y_predicted_logits = criterion_gauss(y_hat, para_merge)
        y_prob = (- y_predicted_logits).exp_()
        # Calculate bpp of z
        z_prob = prob_model(z_hat)
        mse = criterion_mse(x_hat*255., image*255.)
        psnr = 20. * np.log10(255.) - 10 * np.log10(mse.item())
        msssim = criterion_msssim(image*255, x_hat*255)
        bpp_z = - torch.log2(z_prob + 1e-10).sum().item() / num_pixels
        bpp_y = - torch.log2(y_prob).sum().item() / num_pixels
        log.logger.info("%s - PSNR:%.2f, MS-SSIM:%.5f, bpp:%.4f/%.4f"%(img_name,psnr,msssim,bpp_y,bpp_z))
        
        ## Compress
        y_symbol = y_hat.long() + opt.table_range - 1  # modified
        # y_symbol = y_hat.long() + opt.table_range
        _, cy, hy, wy = y_symbol.shape
        z_symbol = z_hat.long() + opt.table_range - 1
        _, cz, hz, wz = z_symbol.shape
        tables = torch.range(-opt.table_range+1, opt.table_range-2).to(device)

        outputfile = "compressed/%s.bin" % img_name
        bitout = arithmeticcoding.BitOutputStream(open(outputfile, "wb"))
        write_int(bitout, [h,w])  # write shape of image and feature
        write_int(bitout, [hy,wy])
        write_int(bitout, [hz,wz])
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

        # Compress z
        pmf_z = prob_model(tables.repeat([1,cz,1,1]))
        encode_channel(pmf_z.squeeze().cpu().numpy(), z_symbol.cpu().numpy(), enc)

        # Compress y
        y_quant = torch.zeros_like(y_hat).to(device)
        for yi in range(hy):
            print(yi, '-', hy)
            for yj in range(wy):
                para1, para2, para3, S, U, R = auto_regressive_model(y_quant, z_feature, criterion_gauss)
                para_merge = torch.cat([para1.reshape(1, opt.num_parameter, opt.last_channels, 1, hy, wy),
                                        para2.reshape(1, opt.num_parameter, opt.last_channels, 1, hy, wy),
                                        para3.reshape(1, opt.num_parameter, opt.last_channels, 1, hy, wy)],
                                        dim=3)
                para_merge = para_merge.reshape(1, -1, hy, wy)

                pmf_y_logits = criterion_gauss(tables.repeat([cy,1,1,1]).permute(3,0,1,2), 
                                               para_merge[:,:,yi:yi+1,yj:yj+1].repeat(opt.table_range*2-2,1,1,1))
                pmf_y = (- pmf_y_logits).exp_().permute(2,3,1,0)  # [hy, wy, c, Lp-2]
                # print(yj, '-', wy)
                encode_channel(pmf_y[0,0].cpu().numpy(), y_symbol[:,:,yi,yj].detach().cpu().numpy(), enc)
                y_quant[:,:,yi,yj] = y_hat[:,:,yi,yj]

        # for yi in range(hy):
        #     print(yi, '-', hy)
        #     for yj in range(wy):
        #         pmf_y_logits = criterion_gauss(tables.repeat([cy,1,1,1]).permute(3,0,1,2), 
        #                                        para_merge[:,:,yi:yi+1,yj:yj+1].repeat(opt.table_range*2-2,1,1,1))
        #         pmf_y = (- pmf_y_logits).exp_().permute(2,3,1,0)  # [hy, wy, c, Lp-2]
        #         # print(yj, '-', wy)
        #         encode_channel(pmf_y[0,0].cpu().numpy(), y_symbol[:,:,yi,yj].detach().cpu().numpy(), enc)
        enc.finish()  # flush any bits to terminate the coding
        bitout.close()

def decompress(shape_num=64):
    if not os.path.exists('./decompressed'):
        os.mkdir('./decompressed')

    # Set module testing
    decode_model.eval()
    hyper_decode_model.eval()
    auto_regressive_model.eval()
    quant_noise.eval()
    decode_model.eval()

    with torch.no_grad():
        # Decompress
        bitin = arithmeticcoding.BitInputStream(open(opt.input_file, "rb"))
        h, w = read_int(bitin, 2)
        y_shape = read_int(bitin, 2)
        z_shape = read_int(bitin, 2)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
        tables = torch.range(-opt.table_range+1, opt.table_range-2).to(device)

        # Decompress z 
        pmf_z = prob_model(tables.repeat([1,opt.hyper_channels,1,1]))
        z_symbol = decode_channel(pmf_z.squeeze().cpu().numpy(), z_shape, dec)
        z_hat = torch.tensor(z_symbol, dtype=torch.float32).unsqueeze(0).to(device) - opt.table_range + 1
        z_feature = hyper_decode_model(z_hat)

        # Regressive model
        hy, wy = y_shape
        y_hat = torch.zeros((1, opt.last_channels, hy, wy)).to(device)
        for yi in range(hy):
            for yj in range(wy):
                para1, para2, para3, _, _, _ = auto_regressive_model(y_hat, z_feature, criterion_gauss)
                para_merge = torch.cat([para1.reshape(1, opt.num_parameter, opt.last_channels, 1, hy, wy),
                                        para2.reshape(1, opt.num_parameter, opt.last_channels, 1, hy, wy),
                                        para3.reshape(1, opt.num_parameter, opt.last_channels, 1, hy, wy)],
                                        dim=3)
                para_merge = para_merge.reshape(1, -1, hy, wy)
                pmf_logits = criterion_gauss(tables.repeat([opt.last_channels,1,1,1]).permute(3,0,1,2),  # [Lp-2, c, 1, 1]
                                             para_merge[:,:,yi:yi+1,yj:yj+1].repeat(opt.table_range*2-2,1,1,1))
                pmf = (- pmf_logits).exp_().permute(2,3,1,0)  # [1, 1, c, Lp-2]
                y_symbol_i_j = decode_channel(pmf[0,0].cpu().numpy(), [1,1], dec)
                y_hat_i_j = torch.tensor(y_symbol_i_j, dtype=torch.float32).to(device) - opt.table_range + 1
                y_hat[0,:,yi,yj] = y_hat_i_j[:,0,0]

        bitin.close()

        x_hat = decode_model(y_hat)
        x_hat = torch.clamp(x_hat, 0., 1.)
        
        # image de-pad
        pad_up = ((shape_num - h % shape_num) % shape_num ) // 2
        pad_left = ((shape_num - w % shape_num) % shape_num ) // 2
        x_hat = x_hat[:, :, pad_up:pad_up+h, pad_left:pad_left+w]

        # Save image
        img_name = opt.input_file.split('/')[-1].split('.bin')[0]
        decompress_img = "decompressed/%s.png" % img_name
        save_image(x_hat[0].clone(), decompress_img)
        print(img_name)
 
def checkpoint(epoch, model_prefix='checkpoint/'):
    if not os.path.exists(model_prefix):
        os.mkdir(model_prefix)
    model_out_path = os.path.join( model_prefix , "model_epoch_{}.pth".format(epoch) )
    if isinstance(encode_model, torch.nn.DataParallel):
        state = {'encode':encode_model.module, 
                 'decode': decode_model.module, 
                 'pencode':hyper_encode_model.module, 
                 'pdecode':hyper_decode_model.module,
                 'prob': prob_model.module,
                 'autoregressive':auto_regressive_model.module
                 }
    else:
        state = {'encode':encode_model, 
                 'decode': decode_model, 
                 'pencode':hyper_encode_model, 
                 'pdecode':hyper_decode_model,
                 'prob': prob_model,
                 'autoregressive':auto_regressive_model
                 }
    torch.save(state, model_out_path)
    log.logger.info("Checkpoint saved to {}".format(model_out_path))

def restore(model_pretrained):  # modified
    log.logger.info("===> Loading pre-trained model: %s" % model_pretrained)
    state = torch.load(model_pretrained, map_location=torch.device('cpu'))
    if(1):
        encode_model.load_state_dict(state['encode'].state_dict())
        decode_model.load_state_dict(state['decode'].state_dict())
        log.logger.info('Load main AE model.')
    if(1):
        hyper_encode_model.load_state_dict(state['pencode'].state_dict())
        hyper_decode_model.load_state_dict(state['pdecode'].state_dict())
        auto_regressive_model.load_state_dict(state['autoregressive'].state_dict())
        prob_model.load_state_dict(state['prob'].state_dict())
        log.logger.info('Load HyperAE and prob model.')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Compression')
    # Model setting
    parser.add_argument('--channels', type=int, default=192, 
                        help='Conv channels per layer.')
    parser.add_argument('--last_channels', type=int, default=384,
                        help='Conv channels of compression feature (y).')
    parser.add_argument('--hyper_channels', type=int, default=192,
                        help='Conv channels of compression feature (z).')
    parser.add_argument("--na", type=str, default="balle2",
                        help="Network architecture")
    parser.add_argument("--mode", type=str, default="train",
                        help="'train', 'test', 'compress', 'decompress'.")
    parser.add_argument("--loss_type", type=str, default="mse",
                        help="loss function : mse, ms-ssim")
    parser.add_argument("--distribution", type=str, default="gauss",
                        help="distribution type: laplace or gauss")
    parser.add_argument("--num_parameter", type=int, default=3,
                        help="distribution parameter num: 1 for sigma, 2 for mean&sigma, 3 for mean&sigma&pi")
    parser.add_argument("--norm", type=str, default="GDN",
                        help="Normalization Type: GDN, GSDN")
    parser.add_argument('--K', type=int, default=1, help='the number of mixed distribution')
    # Data setting
    parser.add_argument('--train_dir', type=str, help='Train image dir.')
    parser.add_argument('--test_dir', type=str, help='Test image dir.')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument("--patchSize", type=int, default=256, help="Training Image size.")
    parser.add_argument('--input_file', type=str, help='File to compress or decompress.')
    # Training setting
    parser.add_argument('--nEpochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
    parser.add_argument('--wd', type=float, default=0., help='Weight Decay. Default=0.')
    parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
    parser.add_argument('--threads', type=int, default=4, help='threads for data loader')
    parser.add_argument('--seed', type=int, default=100001431, help='random seed to use.')
    parser.add_argument('--table_range', type=int, default=128, help='range of feature')
    parser.add_argument('--model_prefix', type=str, default="checkpoint/", help='')
    parser.add_argument('--alpha', type=float, help='weight for reconstruction loss', default=0.01 )
    parser.add_argument('--model_pretrained', type=str, default="", help='pre-trained model')
    parser.add_argument('--epoch_pretrained', type=int, default=0, help='epoch of pre-model')
    opt = parser.parse_args()

    # create log
    log_file = '%s.log' % opt.mode
    log = Logger(filename=os.path.join(opt.model_prefix, log_file), 
                 level='info', 
                 fmt="%(asctime)s - %(message)s")
    log.logger.info(opt)

    # Environment setting
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")

    ## Main Auto-encoder model
    log.logger.info('===> Building model')
    encode_model = Balle2Encoder(opt.channels, opt.last_channels, opt.norm)
    decode_model = Balle2Decoder(opt.channels, opt.last_channels, opt.norm)
    # Quantize Mode
    quant_noise = NoiseQuant(table_range=opt.table_range)
    quant_ste = SteQuant(table_range=opt.table_range)
    # Probability model of hyperprior information
    prob_model = Entropy(opt.channels)
    # hyper model
    hyper_encode_model = HyperEncoder(opt.last_channels, opt.hyper_channels, opt.channels)
    hyper_decode_model = HyperDecoder(opt.hyper_channels, opt.last_channels*2, opt.channels)  # modified
    # Auto Regressive model,  modified
    auto_regressive_model = RefAutoRegressiveModel(
                              cin=opt.last_channels,
                              chyper=opt.last_channels*2,
                              cout=opt.last_channels*opt.K*opt.num_parameter,
                              channels=opt.last_channels*3,
                              bias=True,
                            )
    # Construct Hyperprior Criterion
    criterion_gauss = DiscretizedMixGaussLoss(rgb_scale=False, x_min=-opt.table_range, x_max=opt.table_range-1,
                                              num_p=opt.num_parameter, L=opt.table_range*2)

    # Init modules  # modified
    init_method = xavier_normal_init  # xavier_normal_init
    encode_model.apply(init_method)
    decode_model.apply(init_method)
    hyper_encode_model.apply(init_method)
    hyper_decode_model.apply(init_method)
    auto_regressive_model.apply(init_method)
    log.logger.info(encode_model)
    log.logger.info(decode_model)
    log.logger.info(hyper_encode_model)
    log.logger.info(hyper_decode_model)
    log.logger.info(auto_regressive_model)

    # Load pre-trained model
    if(opt.model_pretrained != ""):
        restore(opt.model_pretrained)

    # GPU setting
    if torch.cuda.device_count() > 1:
        encode_model = nn.DataParallel(encode_model)
        decode_model = nn.DataParallel(decode_model)
        hyper_encode_model = nn.DataParallel(hyper_encode_model)
        hyper_decode_model = nn.DataParallel(hyper_decode_model)
        auto_regressive_model = nn.DataParallel(auto_regressive_model)
        prob_model = nn.DataParallel(prob_model)
        criterion_gauss = nn.DataParallel(criterion_gauss)
    encode_model.to(device)
    decode_model.to(device)
    hyper_encode_model.to(device)
    hyper_decode_model.to(device)
    auto_regressive_model.to(device)
    prob_model.to(device)

    if(opt.mode == "compress"):
        compress()
    elif(opt.mode == "decompress"):
        decompress()
    else:
        # Load train set and test set
        log.logger.info('===> Loading datasets')
        test_set = DatasetFromFolder(opt.test_dir, input_transform=Compose([ToTensor()]))
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
        transform = Compose([RandomCrop(opt.patchSize), RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])   
        train_set = DatasetFromFolder(opt.train_dir, input_transform=transform, cache=False)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        # Optimazers and learning rate adjustment
        base_optimizer = torch.optim.Adam([
                             {'params':encode_model.parameters(), 'lr':opt.lr},
                             {'params':decode_model.parameters(), 'lr':opt.lr}
                             ], eps=1e-8, weight_decay=opt.wd)
        entropy_optimizer = torch.optim.Adam([
                                {'params':hyper_encode_model.parameters(), 'lr':opt.lr},
                                {'params':hyper_decode_model.parameters(), 'lr':opt.lr},
                                {'params':auto_regressive_model.parameters(), 'lr':opt.lr},
                                {'params':prob_model.parameters(), 'lr':opt.lr},
                                ], eps=1e-8, weight_decay=opt.wd)
        lr_step = list(np.linspace(opt.epoch_pretrained, opt.nEpochs, 6, dtype=int))[1:]
        lr_scheduler = LearningRateScheduler(mode='stagedecay',
                                             lr=opt.lr,
                                             num_training_instances=len(train_set),
                                             stop_epoch=opt.nEpochs,
                                             warmup_epoch=opt.nEpochs*0.,  # modified
                                             stage_list=lr_step,
                                             stage_decay=0.75)
        lr_scheduler.update_lr(opt.epoch_pretrained*len(train_set))
        log.logger.info("LR change in:")
        log.logger.info(lr_step)

        # Train or Test
        if(opt.mode == "train"):
            ckpt_stage = list(np.linspace(opt.epoch_pretrained, opt.nEpochs, 6, dtype=int))[1:]
            log.logger.info("Save checkpoint in:")
            log.logger.info(ckpt_stage)
            test(0)
            for epoch in range(opt.epoch_pretrained+1, opt.nEpochs+1):
                train(epoch)
                if epoch%10==0:
                    test(epoch)
                if epoch in ckpt_stage:  # modified
                    checkpoint(epoch, opt.model_prefix)
        elif(opt.mode == "test"):
            test()