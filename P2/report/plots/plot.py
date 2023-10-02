"""
Create plots from the downloaded .csv files.
.csv files were downloaded from wandb.ai
Details for the differen plots can be found in 
DLAD_exercise_2_notes.pdf
"""

import csv
import glob
import os
import matplotlib.pyplot as plt


""" Helper functions. """

def read_csv_data(path, csv_data, data_dict):
    """
    Reads in the relevant csv file columns and stores each column
    in the data_dict dictionary. The column name is used as dict
    key and its value is a list of the column values. 
    ---------
    path: String, filename
    csv_data: Dictionary to store the values of data_dict
    data_dict: Empty dictionary used for storing the column values
    """
    ending = path[-6:-4]
    if ending == 'th':
        prediction = 'depth'
    elif ending == 'gm':
        prediction = 'segm'
    elif ending == 'er':
        prediction = 'grader'
    
    with open(path, newline='') as csvfile:
        tdr = csv.reader(csvfile, delimiter=',')
        # Construct dict keys. 
        indexes = []
        for header_row in tdr:
            # Loop over column headers. 
            cnt = 0
            for header in header_row:
                x = header[-2:]
                if x == 'th' or x == 'eg' or x == 'er':
                    data_dict[cnt] = (header, [])
                    indexes.append(cnt)
                cnt+=1
            break

        # Construct dict values. 
        epochs = 0
        for sample in tdr:
            for idx in indexes:
                if sample[idx] != '':
                    data_dict[idx][1].append(float(sample[idx]))
            epochs+=1
    
    # Store data in csv_data. 
    csv_data[prediction] = data_dict
    if csv_data['epochs'] < epochs:
        csv_data['epochs'] = epochs


def create_store_plot(csv_data, path, fname):
    """
    Creates a line plot with the current .csv file data. 
    Creates one subplot for depth, segmentation, and grader performance. 
    Stores a .png image in the png image folder of the current task. 
    ----------
    csv_data: Dict that has all the data for the plots stored. 
    path: String, path to png folder where the plots are stored.
    fname: String representing the file name.  
    """
    limit = csv_data['epochs']
    x_lim = limit-1

    if (x_lim % 2 == 0) or (x_lim % 3 == 0):
        if x_lim % 2 == 0:
            stepsize = 2
        elif x_lim % 3 == 0:
            stepsize = 3
        x_lim+=1
    else:
        limit_1 = limit + 1
        if limit_1%2 == 0:
            stepsize = 3
        else:
            stepsize = 2
        x_lim = limit_1+1

    if x_lim > 20: stepsize*=2 # Bigger steps for more epochs. 
    x_ticks = [i for i in range(0, x_lim, stepsize)]
    x_max = x_ticks[-1]

    epochs = [i for i in range(0, limit)] # x-axis array. 
    length = len(epochs)
    keys = ['depth', 'segm', 'grader']

    f, plots = plt.subplots(1, 3)
    titles = ['Depth', 'Semseg', 'Grader']
    y_labels = ['SI-lgRMSE [1]', 'IoU [%]', 'Multitask metric [1]']
    
    for k in range(0, 3):
        mx = 0    # Max value of y-axis. 
        mi = 1e9  # Min value of y-axis. 
        
        for i in csv_data[keys[k]].values():
            l = len(i[1])
            if k == 2:
                # Create legend only for one subplot as all subplots store the same information. 
                label = create_legend_label(i[0])
                if l < length:
                    plots[k].plot([i for i in range(l)], i[1], label=label) # , marker='.', markersize='11', markerfacecolor='white')
                else:
                    plots[k].plot(epochs, i[1], label=label) # , marker='.', markersize='11', markerfacecolor='white')
            else:
                if l < length:
                    plots[k].plot([i for i in range(l)], i[1]) # , marker='.', markersize='11', markerfacecolor='white')
                else:
                    plots[k].plot(epochs, i[1]) #, marker='.', markersize='11', markerfacecolor='white')
            m = max(i[1])
            n = min(i[1])
            if m > mx: mx = m 
            if n < mi: mi = n
        
        mx = int(mx + (10 - (int(mx) % 10)))
        mi = int(mi - (int(mi) % 10))

        # Plot settings. 
        plots[k].set_title(titles[k], fontsize=14)
        plots[k].set_xlabel('Epochs [1]', fontsize=14)
        plots[k].set_ylabel(y_labels[k], fontsize=14)
        plots[k].set_xticks(x_ticks, )
        plots[k].set_yticks([i for i in range(mi, mx+1, 10)])
        plots[k].tick_params(labelsize=14)
        plots[k].set_xlim([0, x_max])
        plots[k].set_ylim([mi, mx])

    # Legend. 
    handles, labels = plots[k].get_legend_handles_labels()
    f.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.93), ncol=3, fancybox=False, edgecolor='black', fontsize=12)

    # PDF or PNG
    p = path+fname[0:-9]+'.pdf'
    # p = path+fname[0:-9]+'.png'
    
    # Print info about the file that is currently being plotted. 
    print(fname)

    # plt.show()
    # exit()
    
    f.set_size_inches(16.2*3/2, 10/2)
    plt.savefig(p, bbox_inches='tight', dpi=400)


def create_legend_label(run_name):
    """
    Takes the name of the NN training run as input and creates a label
    for the corresponding plot. 
    ---------
    run_name: String representing the name of the nn training run. 
    """
    run_name = run_name.split('_')[2:-2]
    plot_name = ''
    limit = len(run_name)
    for i in range(limit):
        if run_name[i] == 'adam':
            plot_name += run_name[i].capitalize()
        elif run_name[i] == 'sgd':
            plot_name += run_name[i].capitalize()
        elif run_name[i] == 'lr':
            i+=1
            plot_name += ' LR' + run_name[i]
        elif run_name[i] == 'bs':
            i+=1
            plot_name += ' BS' + run_name[i]
        elif run_name[i] == 'wsegdep':
            plot_name += ' Taskweight (D/S): ' + run_name[i+1] + '/' + run_name[i+2]
            i+=2
        elif run_name[i] == 'pretr':
            i+=1
            if run_name[i][0].lower() == 't':
                plot_name += ' Pretrained'
        elif run_name[i] == 'dilconv':
            i+=1
            if run_name[i][0].lower() == 't':
                plot_name += ' Dilated'  
        elif run_name[i] == 'sa':
            plot_name += ' SA'
        elif run_name[i] == 'branched':
            plot_name += ' Branched'
        elif run_name[i] == 'dcd':
            i+=1
            if run_name[i] == 'deeper':
                plot_name += ' Decoder: 3x2Dconv'
            else:
                plot_name += ' Decoder: 1x2Dconv'
        elif run_name[i] == 'aspp':
            plot_name += ' ASPP'
            i+=1
    return plot_name


""" Create plots and store them in the specified directories. """

if __name__ == '__main__':
     
    num_tasks = 4
    csv_data = {
        'depth': None,
        'segm': None,
        'grader': None,
        'epochs': 0,
    }
    
    # Loop over the task folders.
    for task_idx in range(1, num_tasks):
        # Path to target directory. 
        path = f'./task{task_idx}/csv/'
        store_path = f'./task{task_idx}/png/'
        cnt = 1
        # Loop over all files in the specified directory. 
        for f in os.listdir(path):
            data_dict = {}
            read_csv_data(path+f, csv_data, data_dict)
            
            if cnt == 3:
                # Create plot and store image. 
                create_store_plot(csv_data, store_path, f)
                csv_data['epochs'] = 0
                cnt = 0
            cnt+=1
