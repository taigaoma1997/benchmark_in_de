import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colour
import torch

from sklearn.metrics import r2_score


def compare_param_dist(param_raw, param_pred):
    len_param = len(param_raw)
    category = (['height'] * len_param + ['gap'] * len_param + ['period'] * len_param + ['diameter'] * len_param) * 2
    param_raw = param_raw.T.reshape(-1)
    param_pred = param_pred.T.reshape(-1)
    type = ['raw'] * len(param_pred) + ['pred'] * len(param_pred)
    res_df = pd.DataFrame({'val':np.concatenate((param_raw, param_pred)), 'cat':category, 'type':type})
    sns.boxplot(x='cat', y='val', data=res_df, hue='type')

def compare_cie_dist(cie_raw, cie_pred):
    len_cie = len(cie_raw)
    category = (['x'] * len_cie + ['y'] * len_cie + ['Y'] * len_cie) * 2
    cie_raw = cie_raw.T.reshape(-1)
    cie_pred = cie_pred.T.reshape(-1)
    type = ['raw'] * len(cie_pred) + ['pred'] * len(cie_pred)
    res_df = pd.DataFrame({'val':np.concatenate((cie_raw, cie_pred)), 'cat':category, 'type':type})
    sns.boxplot(x='cat', y='val', data=res_df, hue='type')

def plot_cie(cie_raw, cie_pred):
    from colour.plotting import plot_chromaticity_diagram_CIE1931
    from matplotlib.patches import Polygon

    fig, ax = plot_chromaticity_diagram_CIE1931()
    srgb = Polygon(list(zip([0.64, 0.3, 0.15], [0.33, 0.6, 0.06])), facecolor='0.9', alpha=0.1, edgecolor='k')
    ax.add_patch(srgb)
    ax.scatter(cie_raw[:,0], cie_raw[:,1], s=1, c='b')
    ax.scatter(cie_pred[:,0], cie_pred[:,1], s=1, c='k')
    return fig

def plot_cie_raw_pred_1(cie_raw, cie_pred):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    titles = ['x', 'y', 'Y']
    for i in range(3):
        raw_pred = np.array(sorted(zip(cie_raw[:, i], cie_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1])
        ax[i].plot([raw_pred[:,0].min(), raw_pred[:,0].max()], [raw_pred[:,1].min(), raw_pred[:,1].max()], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
    plt.show()
    
def plot_cie_raw_pred(cie_raw, cie_pred):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    titles = ['x', 'y', 'Y']
    xlim = [[0.1,0.6],[0.0, 0.8],[0.0,0.7]]
    for i in range(3):
        raw_pred = np.array(sorted(zip(cie_raw[:, i], cie_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1], s =3 )
        ax[i].plot(xlim[i],xlim[i], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        ax[i].set_xlim(xlim[i])
        ax[i].set_ylim(xlim[i])
    plt.show()
    
def plot_cie_raw_pred_1(cie_raw, cie_pred):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=150)
    titles = ['x', 'y', 'Y']
    xlim = [[0,1],[0,1],[0,1]]
    for i in range(3):
        raw_pred = np.array(sorted(zip(cie_raw[:, i], cie_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1], s =3 )
        ax[i].plot(xlim[i],xlim[i], c='k')
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        ax[i].set_xlim(xlim[i])
        ax[i].set_ylim(xlim[i])
    plt.show()

def plot_struc_raw_pred(param_raw, param_pred, a = 1):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    titles = ['Height', 'Gap', 'Period','Diamater']
    xlim = [[0, 210], [150, 330], [280, 720], [75, 165]]
    for i in range(4):
        raw_pred = np.array(sorted(zip(param_raw[:, i], param_pred[:, i])))
        ax[i].scatter(raw_pred[:, 0], raw_pred[:, 1])
        
        
        ax[i].set_xlabel('ground truth')
        ax[i].set_ylabel('predicted')
        if a==1:
            ax[i].set_xlim(xlim[i])
            ax[i].set_ylim(xlim[i])
            ax[i].plot(xlim[i],xlim[i], c='k')
            ax[i].set_title(titles[i])
            ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
            continue
        ax[i].set_title(titles[i] + ' (r2 score = {:.3f})'.format(r2_score(raw_pred[:, 0], raw_pred[:, 1])))
    plt.show()
    
def plt_abs_err(CIE_x, cie_pred):
    abs_err = abs(CIE_x - cie_pred)
    abs_mean = sum(abs_err)/len(abs_err)
    
    plt.figure(figsize = [8, 7] , dpi=150)
    plt.subplot(3,1 ,1)
    plt.scatter(CIE_x[:,0],abs_err[:,0], color='r',label='x')
    plt.axhline(y=abs_mean[0],color='r', linestyle='-')
    plt.text(0.5,abs_mean[0] , str(round(abs_mean[0],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.scatter(CIE_x[:,1],abs_err[:,1], color='g',label='y')
    plt.axhline(y=abs_mean[1], color='g',linestyle='-')
    plt.text(0.6,abs_mean[1] , str(round(abs_mean[1],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1, 3)
    plt.scatter(CIE_x[:,2],abs_err[:,2], color='b',label='Y')
    plt.axhline(y=abs_mean[2], color='b',linestyle='-')
    plt.text(0.6,abs_mean[2] , str(round(abs_mean[2],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

def plt_abs_percent_err(CIE_x, cie_pred):
    abs_err = abs(CIE_x - cie_pred)/CIE_x
    abs_mean = sum(abs_err)/len(abs_err)
    
    plt.figure(figsize = [8, 7], dpi=150)
    plt.subplot(3,1 ,1)
    plt.scatter(CIE_x[:,0],abs_err[:,0], color='r',label='x')
    plt.axhline(y=abs_mean[0],color='r', linestyle='-')
    plt.text(0.5,abs_mean[0] , str(round(abs_mean[0],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.scatter(CIE_x[:,1],abs_err[:,1], color='g',label='y')
    plt.axhline(y=abs_mean[1], color='g',linestyle='-')
    plt.text(0.6,abs_mean[1] , str(round(abs_mean[1],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()
    
    plt.subplot(3,1, 3)
    plt.scatter(CIE_x[:,2],abs_err[:,2], color='b',label='Y')
    plt.axhline(y=abs_mean[2], color='b',linestyle='-')
    plt.text(0.6,abs_mean[2] , str(round(abs_mean[2],4)), bbox=dict(facecolor='w', alpha=0.5))
    plt.ylabel('Abs error')
    plt.legend()

    
def plt_hist_struc(param, labels):
    plt.figure(figsize = [20, 3])
    plt.subplot(1, 4,1)
    plt.hist(param[:,0], bins=20, histtype='step', label=labels)
    plt.title('Height histogram')
    plt.xlabel('Height/(nm)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 4,2)
    plt.hist(param[:,1], bins=20, histtype='step', label=labels)
    plt.title('Gap histogram')
    plt.xlabel('Gap/(nm)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.hist(param[:,2], bins=20, histtype='step', label=labels)
    plt.title('Period histogram')
    plt.xlabel('Period/(nm)')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.hist(param[:,3], bins=20, histtype='step', label=labels)
    plt.title('Diameter histogram')
    plt.xlabel('Diamater/(nm)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    

# boxplot of predicted structure
def boxplot_struc_raw_pred(param_raw, param_pred, a = 1):
    fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    titles = ['Height', 'Gap', 'Period','Diamater']
    xlim = [[30, 200], [160, 320], [300, 720], [80, 160]]
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    for i in range(4):
        raw_pred = np.array(sorted(zip(param_raw[:, i], param_pred[:, i])))
        bp = ax[i].boxplot(raw_pred[:, 1], widths=0.4, meanprops=meanpointprops, meanline=False, showmeans=True)
        ax[i].fill_between([0.7, 1.3],[xlim[i][0]], [xlim[i][1]], color='lightgray')
        ax[i].set_xlim([0.7, 1.3])
        ax[i].set_title(titles[i]+'/(nm)')

        
    plt.legend([bp['medians'][0], bp['means'][0],bp['fliers'][0]], ['median', 'mean', 'outliers'], loc='upper right',bbox_to_anchor=(1.0, 1.3))
    plt.show()

# compare the difference of color VS the difference of structures

def struc_vs_color(param_raw, param_pred, cie_raw, cie_pred):
    M = len(param_pred)
    loss_struc = []
    loss_cie = []
    for i in range(M):
        loss_struc.append(np.sum((abs(param_raw[i,:]-param_pred[i,:]))**1))
        loss_cie.append(np.sum((abs(cie_raw[i,:]-cie_pred[i,:]))**1))
    
    #print(loss_cie)
    #print(loss_struc)
    plt.figure(1)
    plt.scatter(loss_struc, loss_cie)
    plt.xlabel('Structure Absolute Difference')
    plt.ylabel('Color Absolute Difference')

def tandem_struc_vs_color(model, forward_model, dataset, show=1):
    # evaluate the tandem prediction using any desired forward model
    # x: structure ; y: CIE 

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        x_pred = model.inverse_model(y, None)
        y_pred = forward_model(x_pred, None)
        
    # get MSE for the design of each model 
    #rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1)).cpu().numpy().tolist()
    #rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1)).cpu().numpy().tolist()
    rmse_design = abs(x_pred - x).sum(dim=1).cpu().numpy().tolist()
    rmse_cie = abs(y_pred - y).sum(dim=1).cpu().numpy().tolist()
    k, b = np.polyfit(rmse_design, rmse_cie, 1)
    x = np.linspace(min(rmse_design), max(rmse_design), 100)
    y = k*x+b
    
    #print(rmse_design, rmse_cie)
    #print(max(rmse_design))
    if show==1:
        fig,ax = plt.subplots()
        plt.scatter(rmse_design, rmse_cie)
        plt.plot(x,y,'r', label="y= {:.3f} x+{:.3f}".format(k,b))
        
        plt.xlabel('Structure Absolute Difference')
        plt.ylabel('Color Absolute Difference')
        plt.legend()
        plt.ylim([-.01, 0.3])
        plt.xlim([0, 3])

        ax2 = ax.twinx()
        plt.hist(rmse_design, density=True, bins=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8], color='b', histtype='step', label='Density of structure diff')
        plt.ylabel('Hist of Structure loss')
        plt.legend(bbox_to_anchor=(1.0, 1.15))
        plt.show()



    return 1

def tandem_hyper_volume(model, forward_model, dataset, show=1, alpha=0.5, ref = [10, 1]):
    # evaluate the tandem prediction using any desired forward model, based on the reference point and alpha 
    # x: structure ; y: CIE 

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        range_, min_ = torch.tensor(dataset.scaler.data_range_).to(DEVICE), torch.tensor(dataset.scaler.data_min_).to(DEVICE)
        x, y = dataset.x.to(DEVICE), dataset.y.to(DEVICE)
        x_dim = x.size()[1]

        x_pred = model.inverse_model(y, None)
        y_pred = forward_model(x_pred, None)

        
    # get MSE for the design of each model 
    #rmse_design = torch.sqrt((x_pred - x).pow(2).sum(dim=1)).cpu().numpy().tolist()
    #rmse_cie = torch.sqrt((y_pred - y).pow(2).sum(dim=1)).cpu().numpy().tolist()
    mae_design = abs(x_pred - x).sum(dim=1).cpu().numpy().tolist()
    print(min(mae_design))
    mae_cie = abs(y_pred - y).sum(dim=1).cpu().numpy().tolist()
    
    k, b = np.polyfit(mae_design, mae_cie, 1)
    x = np.linspace(min(mae_design), max(mae_design), 100)
    y = k*x+b
    
    mae_design = np.array(mae_design)/4.0
    mae_cie = np.array(mae_cie)/3.0

    print('max = ', max(mae_cie))

    if show==1:
        err = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6, 0.65, 0.7, 0.75, 0.8, 1.0]

        err_break = []
        cie_break = []
        cie_mean = []
        cie_err = []
        err_length = []
        
        for i in range(len(err)-1):
            temp = [idx for idx, element in enumerate(mae_design) if (element>err[i]) and (element<=err[i+1])]

            err_break.append(temp)
            cie_break.append(mae_cie[temp])
            cie_mean.append(np.mean(mae_cie[temp]))
            cie_err.append((err[i]+err[i+1])/2)
            err_length.append(len(temp))

        fig,ax = plt.subplots()

        plt.scatter(cie_err, cie_mean)

        i=0

        while ((err_length[i]>0)):

            pos = [(err[i]+err[i+1])/2]
            temp = [idx for idx, element in enumerate(mae_design) if (element>err[i]) and (element<=err[i+1])]
            ax.violinplot(mae_cie[temp], pos, points=100, widths=0.05,showmeans=True, showextrema=True, showmedians=True)

            if i+2>=len(err):
                break
            
            i=i+1

        plt.ylim([-0.001, 0.05])
        plt.xlabel('Mean absolute difference of structure')
        plt.ylabel('Mean absolute difference of CIE')

        ax2 = ax.twinx()
        plt.hist(mae_design,  bins=err, color='b', histtype='step', label='Hist of structure diff')
        
        plt.ylabel('Hist of Structure loss')
        plt.legend(bbox_to_anchor=(1.0, 1.15))
        plt.show()

    mae_design = 1/(alpha+np.array(mae_design))
    #mae_design = 1 - np.array(mae_design)/4.0
    #mae_cie = np.array(mae_cie)/3.0
    des_diff = ref[0] - mae_design
    cie_diff = ref[1] -mae_cie
    a = np.dot(des_diff, cie_diff)/len(cie_diff)
    print('Hyper volume = ',a)

    return a



