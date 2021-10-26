# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-14 14:48:37
import torch
import os
import json
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import interpolate
import scipy.io as scio
import imageio
import cv2
import torch.nn as nn
#from tqdm import tqdm
from scipy.optimize import curve_fit
from imutils import rotate_bound
import warnings
from net.ArbitraryShape import GeneratorNet, SimulatorNet, SimulatorNet_new, SimulatorNet_small, InverseNet_new, cVAE_GSNN, cVAE_hybrid

warnings.filterwarnings('ignore')

current_dir = os.path.abspath(os.path.dirname(__file__))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Binaryloss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x) / x.shape[0]

        return x


class BiBinaryloss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        right = x[:, 0, :, :].masked_fill((x[:, 0, :, :] < 0.5), 1)
        left = x[:, 0, :, :].masked_fill((x[:, 0, :, :] >= 0.5), 0)
        res = torch.mean(1 - right + left)
        return res


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def compute_edge(images, H=64, W=64):
    # compute the gradient of a images
    # ref: https://www.codingame.com/playgrounds/38470/how-to-detect-circles-in-images

    images_fill = np.zeros([H+2, W+2])
    images_fill[1:H+1, 1:W+1] = images
    images_edge = np.zeros([H, W])

    for i in range(1,(H+1),1):
        for j in range(1, (W+1), 1):
            temp_1 = images_fill[i,j]
            temp_2 = images_fill[i,j+1]
            temp_3 = images_fill[i+1,j]
            if (temp_2-temp_1 ==1.0) :
                images_edge[i, j+1] = 1.0
            elif temp_2-temp_1 == -1.0 :
                images_edge[i,j] = 1.0
            if (temp_3-temp_1 == 1.0) :
                images_edge[i+1, j] = 1.0
            elif temp_3-temp_1 == -1.0 :
                images_edge[i,j] = 1.0
            
    edge = []
    for i in range(H):
        for j in range(W):
            if images_edge[i,j] == 1.0:
                edge.append([i,j])

    return edge, images_edge


def circle_rate(edges):
    # judge if a image is circle or not 
    center = np.average(edges, axis=0)
    # print(center)
    dis = []
    for [i,j] in edges:
        dis.append(np.sqrt((i-center[0])**2+(j-center[1])**2))
    # print(np.max(dis), np.min(dis), np.average(dis))
    
    return (np.max(dis)-np.min(dis))/np.average(dis)

def if_circle(err, judge=0.4):
    # return a list under a certain err 
    temp = []
    for i in range(len(err)):
        if err[i]<judge:
            temp.append(i)
    return temp

def delete_err(dataset, err_value=1.0 ):
    #return the deleted dataset with err smaller than err_value
    err = np.zeros([len(dataset), 1])
    for i in range(len(dataset)):
        image = dataset.shape[i,0,:,:]
        edges, edge_imag = compute_edge(image)
        err[i,0] = circle_rate(edges)

    dataset.gap = dataset.gap[err[:,0]<err_value]
    dataset.spectrum = dataset.spectrum[err[:,0]<err_value]
    dataset.shape = dataset.shape[err[:,0]<err_value]

    return dataset

def evaluate_forward_dataset(forward_model, dataset, show = 0, mode=0):
    # To evaluate the forward prediction
    # return: gap_raw, spec_raw, shape_raw, spec_pred
    forward_model.eval()
    with torch.no_grad():
        gap, spectrum_raw, shape_raw = dataset.gap.to(DEVICE), dataset.spectrum.to(DEVICE), dataset.shape.to(DEVICE)

        spectrum_pred = forward_model(shape_raw, gap)
        gap_raw = 200 + gap*200

        criterion = nn.MSELoss()
        rmse_spec = criterion(spectrum_raw, spectrum_pred)

        if show==1:
            print('Spectrum RMSE loss is: {:.6f}.'.format(rmse_spec))
            print('Spectrum MSE loss (based on forward model) is: {:.7f}.'.format(torch.sqrt(rmse_spec)))
    
    return gap_raw.cpu().numpy(), spectrum_raw.cpu().numpy(), shape_raw.cpu().numpy(), spectrum_pred.cpu().numpy()



def evaluate_tandem_accuracy(model, forward_model, dataset, show=1, mode=0):
    # evaluate the tandem prediction using any desired forward model

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        gap, spectrum_raw, shape_raw = dataset.gap.to(DEVICE), dataset.spectrum.to(DEVICE), dataset.shape.to(DEVICE)
        
        gap_raw =  200 + gap*200
        shape_pred, gap_pred = model.pred(spectrum_raw)
        spectrum_pred = forward_model(shape_pred, gap_pred)

        gap_pred = 200 + gap_pred*200
        
        criterion = nn.MSELoss()
        rmse_spec = criterion(spectrum_raw, spectrum_pred)
        
        if show==1:
            print('Spectrum RMSE loss (based on forward model) is: {:.6f}.'.format(rmse_spec))
            print('Spectrum MSE loss (based on forward model) is: {:.7f}.'.format(torch.sqrt(rmse_spec)))
            

    return gap_raw.cpu().numpy(), spectrum_raw.cpu().numpy(), shape_raw.cpu().numpy(), gap_pred.cpu().numpy(), spectrum_pred.cpu().numpy(), shape_pred.cpu().numpy()


def evaluate_tandem_prediction(model, forward_model, spectrum_target, show=1):
    # evaluate the tandem prediction for any desired spectrum using any desired forward model

    model.eval()
    forward_model.eval()
    with torch.no_grad():

        spectrum_target = torch.from_numpy(spectrum_target).float().to(DEVICE)
        shape_pred, gap_pred = model.pred(spectrum_target)

        spectrum_pred = forward_model(shape_pred, gap_pred)

        gap_pred = 200 + gap_pred*200
        

        criterion = nn.MSELoss()
        rmse_spec = criterion(spectrum_target, spectrum_pred)
        
        if show==1:
            print('Spectrum RMSE loss (based on forward model) is: {:.6f}.'.format(rmse_spec))
            print('Spectrum MSE loss (based on forward model) is: {:.7f}.'.format(torch.sqrt(rmse_spec)))
            

    return spectrum_target.cpu().numpy(), gap_pred.cpu().numpy(), spectrum_pred.cpu().numpy(), shape_pred.cpu().numpy()


def evaluate_vae_GSNN_accuracy(vae_GSNN_model, forward_model, dataset, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model
    # x: structure. y: CIE
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    vae_GSNN_model.eval()
    forward_model.eval()
    
    with torch.no_grad():
        
        gap, spec, img = dataset.gap.to(DEVICE), dataset.spectrum.to(DEVICE), dataset.shape.to(DEVICE)

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat = vae_GSNN_model.inference(spec)
        #img_pred, gap_pred, mu, logvar, img_hat, gap_hat = vae_GSNN_model(img, gap, spec)

        spec_pred = forward_model(img_pred, gap_pred)

        gap_pred = 200 + gap_pred*200
        gap = 200 + gap*200

        # get MSE for the design
        criterion = nn.MSELoss()
        rmse_spec = criterion(spec, spec_pred)


        if show==1:
            print('Reconstruct spectrum MSE loss {:.6f}'.format(rmse_spec))
            print('Reconstruct spectrum RMSE loss {:.6f}'.format(torch.sqrt(rmse_spec)))

    return gap.cpu().numpy(), spec.cpu().numpy(), img.cpu().numpy(), gap_pred.cpu().numpy(), spec_pred.cpu().numpy(), img_pred.cpu().numpy()

def evaluate_vae_GSNN_single(vae_GSNN_model, forward_model, spec, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model, and predict 1000 times for a given spec (spec is an array )
    # x: structure. y: CIE
    vae_GSNN_model.eval()
    forward_model.eval()
    
    spec = torch.tensor(spec).float()

    with torch.no_grad():
        
        spec = spec.to(DEVICE)

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat = vae_GSNN_model.inference(spec)
        #img_pred, gap_pred, mu, logvar, img_hat, gap_hat = vae_GSNN_model(img, gap, spec)

        spec_pred = forward_model(img_pred, gap_pred)

        gap_pred = 200 + gap_pred*200


        # get MSE for the design
        criterion = nn.MSELoss()
        rmse_spec = criterion(spec, spec_pred)


        if show==1:
            print('Reconstruct spectrum MSE loss {:.6f}'.format(rmse_spec))
            print('Reconstruct spectrum RMSE loss {:.6f}'.format(torch.sqrt(rmse_spec)))

    return gap_pred.cpu().numpy(), spec_pred.cpu().numpy(), img_pred.cpu().numpy()

def evaluate_cGAN_single(gan_model, forward_model, spec,  configs, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model
    # x: structure. y: CIE
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    gan_model.eval()
    forward_model.eval()
    spec = torch.tensor(spec).float()

    with torch.no_grad():
        
        spec = spec.to(DEVICE)

        z = gan_model.sample_noise(len(spec), configs.prior).to(DEVICE)

        img_pred, gap_pred = gan_model.Generator(spec, z)

        spec_pred = forward_model(img_pred, gap_pred)

        gap_pred = 200 + gap_pred*200

        # get MSE for the design
        criterion = nn.MSELoss()
        rmse_spec = criterion(spec, spec_pred)


        if show==1:
            print('Reconstruct spectrum MSE loss {:.6f}'.format(rmse_spec))
            print('Reconstruct spectrum RMSE loss {:.6f}'.format(torch.sqrt(rmse_spec)))

    return gap_pred.cpu().numpy(), spec_pred.cpu().numpy(), img_pred.cpu().numpy()


def evaluate_cGAN_accuracy(gan_model, forward_model, dataset, configs, show=1):
    # evaluate both the vae_GSNN and vae_hybrid model using a forward model
    # x: structure. y: CIE
    '''
    returns:
        y_raw: original desired xyY
        y_raw_pred: xyY predicted by the forward module for the inversely designed structure
        x_raw: original structure parameters
        x_raw_pred: inversely designed parameters.
    '''
    gan_model.eval()
    forward_model.eval()
    
    with torch.no_grad():
        
        gap, spec, img = dataset.gap.to(DEVICE), dataset.spectrum.to(DEVICE), dataset.shape.to(DEVICE)

        z = gan_model.sample_noise(len(gap), configs.prior).to(DEVICE)

        img_pred, gap_pred = gan_model.Generator(spec, z)

        spec_pred = forward_model(img_pred, gap_pred)

        gap_pred = 200 + gap_pred*200
        gap = 200 + gap*200

        # get MSE for the design
        criterion = nn.MSELoss()
        rmse_spec = criterion(spec, spec_pred)


        if show==1:
            print('Reconstruct spectrum MSE loss {:.6f}'.format(rmse_spec))
            print('Reconstruct spectrum RMSE loss {:.6f}'.format(torch.sqrt(rmse_spec)))

    return gap.cpu().numpy(), spec.cpu().numpy(), img.cpu().numpy(), gap_pred.cpu().numpy(), spec_pred.cpu().numpy(), img_pred.cpu().numpy()




def save_checkpoint(state, path, name):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)

    torch.save(state, name)

    print('Model saved')


def load_checkpoint(path, net, optimizer):
    if not os.path.exists(path):
        raise("File doesn't exist {}".format(path))

    if torch.cuda.is_available():
        state = torch.load(path, map_location='cuda:0')
    else:
        state = torch.load(path, map_location='cpu')
    net.load_state_dict(state['net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optim_state_dict'])

    print('Model loaded')


def interploate(org_data, points=1000):
    org = np.linspace(400, 680, len(org_data))
    new = np.linspace(400, 680, points)
    inter_func = interpolate.interp1d(org, org_data, kind='cubic')
    return inter_func(new)


def make_figure_dir():
    os.makedirs(current_dir + '/figures/loss_curves', exist_ok=True)
    os.makedirs(current_dir + '/figures/test_output', exist_ok=True)


def plot_single_part(wavelength, spectrum, name, legend='spectrum', interpolate=True):
    save_dir = os.path.join(current_dir, 'figures/test_output', name)
    plt.figure()
    plt.plot(wavelength, spectrum, 'ob')
    plt.grid()
    if interpolate:
        new_spectrum = interploate(spectrum)
        new_wavelength = interploate(wavelength)
        plt.plot(new_wavelength, new_spectrum, '-b')
    plt.title(legend)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(legend)
    plt.ylim((0, 1))
    plt.savefig(save_dir)
    plt.close()


def plot_triple_parts(wavelength, clear, dim, real, name, interpolate=True):
    save_dir = os.path.join(current_dir, 'figures/test_output', name)
    plt.figure()
    plt.plot(wavelength, real, 'ob')
    plt.plot(wavelength, clear, 'or')
    plt.plot(wavelength, dim, 'og')
    plt.grid()
    if interpolate:
        new_real = interploate(real)
        new_clear = interploate(clear)
        new_dim = interploate(dim)
        new_wavelength = interploate(wavelength)
        plt.plot(new_wavelength, new_real, '-b', label='RCWA Ground Truth')
        plt.plot(new_wavelength, new_clear, '-r', label='Simulator Clear')
        plt.plot(new_wavelength, new_dim, '-g', label='Simulator Dim')
    plt.title('Comparison of inputs are binary or not')
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transimttance')
    plt.ylim((0, 1))
    plt.savefig(save_dir)
    plt.close()


def plot_both_parts(wavelength, real, fake, name, legend='Real and Fake', interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Real', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Real')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Fake', color=color_right)  # we already handled the x-label with ax1
    ax2.plot(wavelength, fake, 'o', color=color_right, label='Fake')
    if interpolate:
        new_fake = interploate(fake)
        ax2.plot(new_wavelength, new_fake, color=color_right)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    plt.ylim((0, 1))
    plt.title(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def plot_both_parts_2(wavelength, real, cv, name, legend='Spectrum and Contrast Vector', interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transimttance', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Spectrum')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    # ax1.legend()
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()
    plt.ylim((0, 1))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Contrast', color=color_right)  # we already handled the x-label with ax1
    ax2.step(np.linspace(400, 680, 8), np.append(cv, cv[-1]), where='post', color=color_right, label='Contrast Vector')
    # ax2.legend()
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    # plt.ylim((0, 1))
    plt.title(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def make_gif(path):
    images = []
    filenames = sorted((fn for fn in os.listdir(path) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave('tendency.gif', images, duration=0.1)


def rename_all_files(path):
    filelist = os.listdir(path)
    count = 0
    for file in filelist:
        print(file)
    for file in filelist:
        Olddir = os.path.join(path, file)
        if os.path.isdir(Olddir):
            rename_all_files(Olddir)
            continue
        filetype = os.path.splitext(file)[1]
        filename = os.path.splitext(file)[0]
        Newdir = os.path.join(path, str(int(filename)).zfill(4) + filetype)
        os.rename(Olddir, Newdir)
        count += 1


def rect2polar(real, imag):
    complex_number = complex(real, imag)
    return abs(complex_number), cmath.phase(complex_number)


def polar2rect(modu, phase):
    complex_number = cmath.rect(modu, phase)
    return complex_number.real, complex_number.imag


def rect2polar_parallel(real_que, imag_que):
    assert len(real_que) == len(imag_que), "Size mismatch"
    modu_que, phase_que = np.zeros(len(real_que)), np.zeros(len(real_que))
    for i, real, imag in zip(range(len(real_que)), real_que, imag_que):
        modu_que[i], phase_que[i] = rect2polar(real, imag)
    return modu_que, phase_que


def polar2rect_parallel(modu_que, phase_que):
    assert len(modu_que) == len(phase_que), "Size mismatch"
    real_que, imag_que = np.zeros(len(modu_que)), np.zeros(len(modu_que))
    for i, modu, phase in zip(range(len(modu_que)), modu_que, phase_que):
        real_que[i], imag_que[i] = polar2rect(modu, phase)
    return real_que, imag_que


def find_spectrum(thickness, radius, gap, TT_array):
    rows, _ = TT_array.shape
    wavelength, spectrum = [], []
    for row in range(rows):
        if TT_array[row, 1] == thickness and TT_array[row, 2] == radius and TT_array[row, 3] == gap:
            wavelength.append(TT_array[row, 0])
            spectrum.append(TT_array[row, -1])
        else:
            continue
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)
    index_order = np.argsort(wavelength)
    return wavelength[index_order], spectrum[index_order]


def load_mat(path):
    variables = scio.whosmat(path)
    target = variables[0][0]
    data = scio.loadmat(path)
    TT_array = data[target]
    TT_list = TT_array.tolist()
    return TT_list, TT_array


def data_pre(list_all, wlimit):
    dtype = [('wave_length', int), ('thickness', int), ('radius', int), ('gap', int), ('efficiency', float)]
    values = [tuple(single_device) for single_device in list_all]
    array_temp = np.array(values, dtype)
    array_all = np.sort(array_temp, order=['thickness', 'radius', 'gap', 'wave_length'])

    thickness_list = np.unique(array_all['thickness'])
    radius_list = np.unique(array_all['radius'])
    gap_list = np.unique(array_all['gap'])
    reformed = []

    for thickness in thickness_list:
        for radius in radius_list:
            for gap in gap_list:
                pick_index = np.intersect1d(np.argwhere(array_all['radius'] == radius), np.argwhere(
                    array_all['thickness'] == thickness))
                pick_index = np.intersect1d(pick_index, np.argwhere(array_all['gap'] == gap))
                picked = array_all[pick_index]
                picked = np.sort(picked, order=['wave_length'])
                cur_ref = [thickness, radius, gap]
                for picked_single in picked:
                    cur_ref.append(picked_single[4])

                reformed.append(cur_ref)

    return np.array(reformed), array_all


def inter(inputs, device):
    inputs_inter = torch.ones(inputs.shape[0], inputs.shape[1], 224)
    x = np.linspace(0, 223, num=inputs.shape[2])
    new_x = np.linspace(0, 223, num=224)

    for index_j, j in enumerate(inputs):
        for index_jj, jj in enumerate(j):
            y = jj
            f = interpolate.interp1d(x, y, kind='cubic')
            jj = f(new_x)
            inputs_inter[index_j, index_jj, :] = torch.from_numpy(jj)

    inputs_inter = inputs_inter.double().to(device)

    return inputs_inter


def RCWA_parallel(eng, w_list, thick_list, r_list, gap, acc=5):
    import matlab.engine
    batch_size = len(thick_list)
    spec = np.ones((batch_size, len(w_list)))
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    for i in range(batch_size):
        thick = thick_list[i]
        thick = matlab.double([thick])
        r = r_list[i]
        r = matlab.double([r])
        for index, w in enumerate(w_list):
            w = matlab.double([w])
            spec[i, index] = eng.RCWA_solver(w, gap, thick, r, acc)

    return spec


def RCWA(eng, w_list, thick, r, gap, acc=5, medium=1, shape=0):
    import matlab.engine
    spec = np.ones(len(w_list))
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    thick = matlab.double([thick])
    r = matlab.double([r])
    medium = matlab.double([medium])
    shape = matlab.double([shape])
    for index, w in enumerate(w_list):
        w = matlab.double([w])
        spec[index] = eng.RCWA_solver(w, gap, thick, r, acc, medium, shape)

    return spec


def RCWA_arbitrary(eng, gap, img_path, thickness=500, acc=5):
    import matlab.engine
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    thick = matlab.double([thickness])
    spec = eng.cal_spec(gap, thick, acc, img_path, nargout=2)
    spec_TE, spec_TM = spec
    return spec_TE, spec_TM


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def keep_range(data, low=0, high=1):
    index = np.where(data > high)
    data[index] = high
    index = np.where(data < low)
    data[index] = low

    return data


def gauss_spec_valley(f, mean, var, depth=0.2):
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def gauss_spec_peak(f, mean, var, depth=0.2):
    return (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_gauss_spec(f):
    depth = np.random.uniform(low=0.0, high=0.05)
    mean = np.random.uniform(low=400, high=600)
    var = np.random.uniform(low=20, high=40)
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_step_spec(f):
    depth = np.random.uniform(low=0.0, high=0.2)
    duty_ratio = np.random.uniform(low=0.1, high=0.3)
    width = int(len(f) * duty_ratio / 2)
    valley = np.random.randint(low=len(f) / 3, high=2 * len(f) / 3, dtype=int)
    spec = np.random.uniform(low=0.7, high=1, size=len(f))
    spec[valley - width:valley + width] = depth
    return spec


def random_gauss_spec_combo(f, valley_num):
    spec = np.zeros(len(f))
    for i in range(valley_num):
        spec += random_gauss_spec(f)
    return normalization(spec)


def spec_jitter(spec, amp):
    return normalization(spec + np.random.uniform(low=-amp, high=amp, size=spec.size))


def gauss10(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9):

    return a0 * np.exp(-((x - m0) / s0)**2) + a1 * np.exp(-((x - m1) / s1)**2) + a2 * np.exp(-((x - m2) / s2)**2) + \
        a3 * np.exp(-((x - m3) / s3)**2) + a4 * np.exp(-((x - m4) / s4)**2) + a5 * np.exp(-((x - m5) / s5)**2) + \
        a6 * np.exp(-((x - m6) / s6)**2) + a7 * np.exp(-((x - m7) / s7)**2) + a8 * np.exp(-((x - m8) / s8)**2) + \
        a9 * np.exp(-((x - m9) / s9)**2)


def gauss10_tensor(in_tensor):
    wave_tensor = torch.range(400, 680, 10)

    def gauss_tensor(wave_tensor, a, m, s):
        return a * torch.exp(-torch.pow((wave_tensor - m) / s, 2))

    out_tensor = torch.zeros_like(wave_tensor)
    for i in range(10):
        out_tensor += gauss_tensor(wave_tensor, in_tensor[i], in_tensor[i + 10], in_tensor[i + 20])
    return out_tensor


def gauss10_curve_fit(spec):
    a_min = [0] * 10
    a_max = [1] * 10
    m_min = [400] * 10
    m_max = [680] * 10
    s_min = [0] * 10
    s_max = [1000] * 10
    min_limit = a_min + m_min + s_min
    max_limit = a_max + m_max + s_max
    wavelength = np.linspace(400, 680, 29)
    popt, _ = curve_fit(gauss10, wavelength, spec, bounds=(min_limit, max_limit))
    return popt


def cal_contrast(wavelength, spec, spec_start, spec_end):
    spec_range_in = spec[np.argwhere((wavelength <= spec_end) & (wavelength >= spec_start))]
    sepc_range_out = spec[np.argwhere((wavelength > spec_end) | (wavelength < spec_start))]
    contrast = np.max(spec_range_in) / np.max(sepc_range_out)
    return contrast


def cal_contrast_vector(spec):
    wavelength = np.linspace(400, 680, 29)
    contrast_vector = np.zeros(7)
    for i in range(len(contrast_vector)):
        contrast_vector[i] = cal_contrast(wavelength, spec, 400 + i * 40, 440 + i * 40)
    return contrast_vector


def plot_possible_spec(spec):
    min_index = np.argmin(spec, axis=1)
    min_sort = np.argsort(min_index)
    TE_spec = spec[min_sort, :]
    wavelength = np.linspace(400, 680, 29)
    # TE_spec = cv2.resize(src=TE_spec, dsize=(1000, 1881), interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.pcolor(TE_spec, cmap=plt.cm.jet)
    plt.xlabel('Wavelength (nm)')
    # plt.xlabel('Index of elements')
    plt.ylabel('Index of Devices')
    plt.title('Possible Contrast Distribution (TM)')
    # plt.title('Gaussian Amplitude after Decomposition')
    # plt.title('Possible Spectrums of Arbitrary Shapes (' + title + ')')
    # plt.title(r'Possible Spectrums of Square Shape ($T_iO_2$)')
    # plt.xticks(np.arange(len(wavelength), step=4), np.uint16(wavelength[::4]))
    plt.xticks(np.arange(8), np.uint16(wavelength[::4]))
    plt.yticks([])
    cb = plt.colorbar()
    cb.ax.set_ylabel('Contrast')
    plt.show()


def data_pre_arbitrary(T_path):
    print("Waiting for Data Preparation...")
    _, TT_array = load_mat(T_path)
    all_num = TT_array.shape[0]
    print(all_num)
    all_name_np = TT_array[:, 0]
    all_gap_np = (TT_array[:, 1] - 200) / 200
    all_spec_np = TT_array[:, 2:]
    # all_gauss_np = np.zeros((all_num, 60))
    all_shape_np = np.zeros((all_num, 1, 64, 64))
    all_ctrast_np = np.zeros((all_num, 14))
    with tqdm(total=all_num, ncols=70) as t:
        delete_list = []
        for i in range(all_num):
            # shape
            find = False
            name = str(int(all_name_np[i]))
            filelist = os.listdir('polygon')
            for file in filelist:
                if name == file.split('_')[0]:
                    img_np = cv2.imread('polygon/' + file, cv2.IMREAD_GRAYSCALE)
                    all_shape_np[i, 0, :, :] = img_np / 255
                    find = True
                    break
                else:
                    continue
            if not find:
                print("NO match with " + str(i) + ", it will be deleted later!")
                delete_list.append(i)
            # calculate contrast
            all_ctrast_np[i, :] = np.concatenate((cal_contrast_vector(
                all_spec_np[i, :29]), cal_contrast_vector(all_spec_np[i, 29:])))
            # gauss curve fit
            # try:
            #     all_gauss_np[i, :] = np.concatenate(
            #         (np.array(gauss10_curve_fit(all_spec_np[i, :29])), np.array(gauss10_curve_fit(all_spec_np[i, 29:]))))
            # except:
            #     print("Optimal parameters not found with " + str(i) + ", it will be deleted later!")
            #     if find:
            #         delete_list.append(i)
            t.update()
    # delete error guys
    all_name_np = np.delete(all_name_np, delete_list, axis=0)
    all_gap_np = np.delete(all_gap_np, delete_list, axis=0)
    all_spec_np = np.delete(all_spec_np, delete_list, axis=0)
    all_shape_np = np.delete(all_shape_np, delete_list, axis=0)
    # all_gauss_np = np.delete(all_gauss_np, delete_list, axis=0)
    all_ctrast_np = np.delete(all_ctrast_np, delete_list, axis=0)
    np.save('data/all_gap.npy', all_gap_np)
    np.save('data/all_spec.npy', all_spec_np)
    np.save('data/all_shape.npy', all_shape_np)
    np.save('data/all_ctrast.npy', all_ctrast_np)
    print("Data Preparation Done! All get {} elements!".format(all_num - len(delete_list)))


def data_enhancement():
    
    # add extra data: rotate the images 
    
    print("Waiting for Data Enhancement...")
    all_gap_org = np.load('data/all_gap.npy')
    all_spec_org = np.load('data/all_spec.npy')
    all_shape_org = np.load('data/all_shape.npy')
    all_spec_90_270 = np.zeros_like(all_spec_org)
    all_shape_90, all_shape_270, all_shape_180 = np.zeros_like(
        all_shape_org), np.zeros_like(all_shape_org), np.zeros_like(all_shape_org)
    with tqdm(total=all_gap_org.shape[0], ncols=70) as t:
        for i in range(all_gap_org.shape[0]):
            all_spec_90_270[i, :] = np.concatenate((all_spec_org[i, 29:], all_spec_org[i, :29]))
            all_shape_90[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 90)
            all_shape_180[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 180)
            all_shape_270[i, 0, :, :] = rotate_bound(all_shape_org[i, 0, :, :], 270)
            t.update()
    all_gap_en = np.concatenate((all_gap_org, all_gap_org, all_gap_org, all_gap_org), axis=0)
    all_spec_en = np.concatenate((all_spec_org, all_spec_90_270, all_spec_org, all_spec_90_270), axis=0)
    all_shape_en = np.concatenate((all_shape_org, all_shape_90, all_shape_180, all_shape_270), axis=0)

    all_num = all_gap_en.shape[0]
    permutation = np.random.permutation(all_num).tolist()
    all_gap_en, all_spec_en, all_shape_en = all_gap_en[permutation], all_spec_en[permutation, :], all_shape_en[permutation, :, :, :]

    np.save('data/all_gap_en.npy', all_gap_en)
    np.save('data/all_spec_en.npy', all_spec_en)
    np.save('data/all_shape_en.npy', all_shape_en)
    print('Data Enhancement Done!')

def count_params(model):

    return sum([np.prod(layer.size()) for layer in model.parameters() if layer.requires_grad])

def mask(output_shapes, output_gaps):
    batch_size = output_gaps.shape[0]
    mask = torch.zeros_like(output_shapes).float()
    for i in range(batch_size):
        bound = int(output_shapes.shape[-1] // (int(output_gaps[i, 0] * 200 + 200) / 20) + 1)
        mask[i, 0, bound:-bound, bound:-bound] = 1
    output_shapes = output_shapes * mask
    return output_shapes


def center_comp(ori):
    global mask
    mask = torch.zeros_like(ori)
    global search_or_not
    search_or_not = torch.zeros_like(ori)
    global img
    img = ori

    for i in range(img.shape[0]):
        center_x, center_y = find_center(img[i, 0])
        assert img[i, 0, center_x, center_y] > 0
        mask[i, 0, center_x, center_y] = 1

    for bs_no in range(img.shape[0]):
        flag = search(bs_no, center_x, center_y)

    return img * mask


def search(bs_no, cor_x, cor_y):
    if cor_x < 0 or cor_x >= img.shape[2] or cor_y < 0 or cor_y >= img.shape[3]:
        return -1

    if search_or_not[bs_no, 0, cor_x, cor_y] == 0:
        search_or_not[bs_no, 0, cor_x, cor_y] = 1
    else:
        return -1

    if img[bs_no, 0, cor_x, cor_y] == 1:
        mask[bs_no, 0, cor_x, cor_y] = 1
        flag_left = search(bs_no, cor_x - 1, cor_y)
        flag_right = search(bs_no, cor_x + 1, cor_y)
        flag_up = search(bs_no, cor_x, cor_y - 1)
        flag_down = search(bs_no, cor_x, cor_y + 1)
    else:
        mask[bs_no, 0, cor_x, cor_y] = 0

    return 0


def find_center(img):
    center_x = img.shape[0] // 2
    center_y = img.shape[1] // 2
    list_point = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            point = (abs(i - center_x) + abs(j - center_y), i, j)
            list_point.append(point)
    list_point.sort(key=takeFirst)
    for i in list_point:
        if img[i[1], i[2]] > 0:
            center_x = i[1]
            center_y = i[2]
            break

    return center_x, center_y


def takeFirst(elem):
    return elem[0]


def binary(output_shapes_org):
    mask_1 = output_shapes_org > 0.5
    mask_0 = output_shapes_org <= 0.5
    output_shapes_new = output_shapes_org.masked_fill(mask_1, 1)
    output_shapes_org = output_shapes_new.masked_fill(mask_0, 0)
    return output_shapes_org


if __name__ == '__main__':
    rename_all_files('results')
